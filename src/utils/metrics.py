from __future__ import annotations

import torch
from torch import Tensor
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: Tensor, target: Tensor) -> float:
    pred_np = _tensor_to_image(pred)
    target_np = _tensor_to_image(target)
    return float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))


def ssim(pred: Tensor, target: Tensor) -> float:
    pred_np = _tensor_to_image(pred)
    target_np = _tensor_to_image(target)
    return float(
        structural_similarity(
            target_np,
            pred_np,
            channel_axis=-1,
            data_range=1.0,
            win_size=11,
        )
    )


def _tensor_to_image(t: Tensor) -> Tensor:
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 4:
        t = make_grid(t, nrow=t.shape[0]).permute(1, 2, 0)
    else:
        t = t.permute(1, 2, 0)
    return t.numpy()

