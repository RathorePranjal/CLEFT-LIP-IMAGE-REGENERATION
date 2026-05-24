from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.models.unet import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cleft-lip inpainting on a single image + mask pair.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--image", type=str, required=True, help="Path to the original RGB image.")
    parser.add_argument("--mask", type=str, required=True, help="Path to the grayscale mask image.")
    parser.add_argument("--output", type=str, required=True, help="Where to save the regenerated PNG.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize target used during training.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def load_inputs(image_path: Path, mask_path: Path) -> Tuple[Image.Image, Image.Image]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    return img, mask


def preprocess(
    image: Image.Image,
    mask: Image.Image,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    img_t = image_tf(image)
    mask_t = mask_tf(mask)
    mask_bin = (mask_t > 0.5).float()
    masked_img = img_t * (1.0 - mask_bin)
    model_in = torch.cat([masked_img, mask_bin], dim=0).unsqueeze(0)
    return model_in, mask_bin, img_t


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    image, mask = load_inputs(Path(args.image), Path(args.mask))
    model_in, _, _ = preprocess(image, mask, args.image_size)
    model = load_model(args.weights, device)
    output = model(model_in.to(device)).squeeze(0)
    out_img = tensor_to_pil(output)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"Saved regenerated image to {out_path}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

