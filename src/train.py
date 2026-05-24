from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm

from src.data.dataset import MaskedCelebADataset, create_splits
from src.models.unet import InpaintingUNet
from src.utils.metrics import psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mask-aware U-Net on CelebA.")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--mask-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (256=fast, 512=sharper but slower)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--use-perceptual", action="store_true", help="Use perceptual loss for sharper results (recommended)")
    parser.add_argument("--lambda-perceptual", type=float, default=0.1)
    parser.add_argument("--lambda-mask", type=float, default=10.0, help="Weight for loss in masked region (higher = more focus on regenerated area)")
    return parser.parse_args()


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.slice = nn.Sequential(*list(vgg.children())[:16])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(self.slice(pred), self.slice(target))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    l1_loss: nn.Module,
    perceptual: nn.Module | None,
    lambda_perc: float,
    lambda_mask: float,
    log_every: int,
) -> float:
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(inputs)
            # Mask-weighted loss: higher weight in masked region
            pixel_loss = nn.functional.l1_loss(output, target, reduction="none")
            mask_weight = 1.0 + lambda_mask * mask  # Higher weight in masked area
            loss = (pixel_loss * mask_weight).mean()
            
            if perceptual is not None:
                # Perceptual loss helps preserve texture and detail
                perc_loss = perceptual(output, target)
                loss = loss + lambda_perc * perc_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        if step % log_every == 0:
            print(f"step {step}: loss {loss.item():.4f}")
    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    l1_loss: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    psnr_scores, ssim_scores = [], []
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        output = model(inputs)
        total_loss += l1_loss(output, target).item()
        psnr_scores.append(psnr(output, target))
        ssim_scores.append(ssim(output, target))
    avg_loss = total_loss / len(loader)
    return avg_loss, float(sum(psnr_scores) / len(psnr_scores)), float(sum(ssim_scores) / len(ssim_scores))


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    dataset = MaskedCelebADataset(args.image_root, args.mask_root, args.image_size)
    train_ds, val_ds = create_splits(dataset, val_ratio=args.val_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = InpaintingUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    l1_loss = nn.L1Loss()
    perceptual = PerceptualLoss().to(device) if args.use_perceptual else None

    start_epoch = 1
    best_loss = float("inf")

    save_dir = Path(args.save_dir)
    
    # Auto-resume from latest checkpoint if no --resume specified
    if not args.resume:
        checkpoint_files = sorted(save_dir.glob("epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
            args.resume = str(latest_checkpoint)
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("metrics", {}).get("val_loss", best_loss)
        print(f"Resumed from epoch {start_epoch - 1}, continuing from epoch {start_epoch}")

    best_path = Path("artifacts/best.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            l1_loss,
            perceptual,
            args.lambda_perceptual,
            args.lambda_mask,
            args.log_every,
        )
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device, l1_loss)
        print(f"val_loss={val_loss:.4f}, psnr={val_psnr:.2f}, ssim={val_ssim:.3f}")
        metrics = {"train_loss": train_loss, "val_loss": val_loss, "psnr": val_psnr, "ssim": val_ssim}
        save_checkpoint(epoch, model, optimizer, save_dir / f"epoch_{epoch:03d}.pt", metrics)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(epoch, model, optimizer, best_path, metrics)
            print(f"New best model saved to {best_path}")


if __name__ == "__main__":
    main()

