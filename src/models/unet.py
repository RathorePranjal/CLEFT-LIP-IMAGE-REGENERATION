from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))
            layers.insert(-1, nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch // 2, in_ch // 2, 1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class InpaintingUNet(nn.Module):
    """
    Four-channel U-Net where the 4th channel is the binary mask.
    """

    def __init__(self, in_channels: int = 4, bilinear: bool = True, base_ch: int = 64) -> None:
        super().__init__()
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)
        self.outc = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.final_act(x)


def load_model(weights_path: str | Path, device: torch.device) -> InpaintingUNet:
    model = InpaintingUNet()
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

