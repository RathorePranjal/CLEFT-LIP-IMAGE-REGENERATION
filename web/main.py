from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.models.unet import load_model

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "web" / "static"
TEMPLATES_DIR = ROOT_DIR / "web" / "templates"
MODEL_WEIGHTS = ROOT_DIR / "artifacts" / "best.pt"
IMAGE_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None


def get_model() -> torch.nn.Module:
    global MODEL
    if MODEL is None:
        if not MODEL_WEIGHTS.exists():
            raise RuntimeError(
                f"Missing weights at {MODEL_WEIGHTS}. Train a model and copy the best checkpoint to this path."
            )
        MODEL = load_model(MODEL_WEIGHTS, DEVICE)
    return MODEL


def read_image(file: UploadFile, mode: str) -> Image.Image:
    try:
        data = file.file.read()
        if not data:
            raise ValueError("Empty file upload.")
        return Image.open(BytesIO(data)).convert(mode)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}") from exc
    finally:
        file.file.close()


def preprocess(image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    img_t = image_tf(image)
    mask_t = mask_tf(mask)
    mask_bin = (mask_t > 0.5).float()
    masked_img = img_t * (1.0 - mask_bin)
    model_in = torch.cat([masked_img, mask_bin], dim=0).unsqueeze(0)
    return model_in, img_t


def tensor_to_base64(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().clamp(0, 1)
    pil_img = transforms.ToPILImage()(tensor)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


app = FastAPI(title="Cleft Lip Regenerator", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def add_surgical_scar(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Adds a realistic surgical scar overlay to the regenerated area.
    The scar appears as a vertical line in the center of the masked region,
    simulating a post-surgical cleft lip repair scar.
    """
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    
    height, width = mask_array.shape
    
    # Find the center of the masked region
    y_coords, x_coords = np.where(mask_array > 0.5)
    
    if len(x_coords) == 0:
        return image  # No mask, return original
    
    # Get the center x-coordinate and vertical span of the mask
    center_x = int(np.mean(x_coords))
    min_y = int(np.min(y_coords))
    max_y = int(np.max(y_coords))
    
    # Create scar mask (where scar will be drawn)
    scar_mask = np.zeros((height, width), dtype=np.float32)
    
    # Scar properties
    scar_width_base = max(1, max(2, width // 150))  # Base width scales with image size
    
    # Draw scar line with slight natural variation
    for y in range(min_y, max_y):
        if mask_array[y, center_x] > 0.3:  # Only in masked area
            # Add slight horizontal variation (natural scar curvature)
            offset = int(1.5 * np.sin((y - min_y) * 0.08))  # Subtle wave
            x_pos = center_x + offset
            
            if 0 <= x_pos < width:
                # Variable width (slightly thicker in middle, thinner at edges)
                progress = (y - min_y) / max(1, max_y - min_y)
                width_factor = 1.0 - 0.2 * abs(progress - 0.5) * 2
                scar_width = max(1, int(scar_width_base * width_factor))
                
                # Draw scar line
                for dx in range(-scar_width, scar_width + 1):
                    x = x_pos + dx
                    if 0 <= x < width and mask_array[y, x] > 0.3:
                        # Gaussian falloff for smooth edges
                        dist = abs(dx) / max(1, scar_width)
                        intensity = np.exp(-dist * dist * 2)
                        scar_mask[y, x] = max(scar_mask[y, x], intensity)
    
    # Apply scar effect: slightly lighter with pinkish/whitish tint (healed scar appearance)
    scar_intensity = 0.12  # Subtle but visible
    scar_color_shift = np.array([1.08, 1.02, 0.98])  # Slightly pinkish-white
    
    # Expand scar mask to 3 channels
    scar_mask_3d = np.stack([scar_mask] * 3, axis=2)
    
    # Apply scar: blend scar color with image
    scar_effect = img_array * (1.0 - scar_mask_3d * scar_intensity) + \
                  img_array * scar_color_shift * scar_mask_3d * scar_intensity
    
    # Add subtle texture variation to scar (healed skin texture)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.float32)
    scar_effect = np.clip(scar_effect + noise * scar_mask_3d * 0.3, 0, 255)
    
    # Only apply scar in masked regions
    mask_3d = np.stack([mask_array] * 3, axis=2)
    final = img_array * (1.0 - mask_3d) + scar_effect * mask_3d
    
    return Image.fromarray(np.clip(final, 0, 255).astype(np.uint8))


@app.post("/api/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)) -> JSONResponse:
    try:
        base_img = read_image(image, "RGB")
        mask_img = read_image(mask, "L")
        
        # Store original dimensions
        original_size = base_img.size  # (width, height)
        
        # Resize mask to match original image size for blending
        mask_resized = mask_img.resize(original_size, Image.Resampling.NEAREST)
        
        model_in, _ = preprocess(base_img, mask_img)
        model = get_model()
        with torch.no_grad():
            output = model(model_in.to(DEVICE)).squeeze(0)
        
        # Convert tensor to PIL Image
        output_pil = transforms.ToPILImage()(output.detach().cpu().clamp(0, 1))
        
        # Resize output back to original image dimensions
        output_resized = output_pil.resize(original_size, Image.Resampling.BICUBIC)
        
        # Blend output with original: use model output only in masked areas, keep original elsewhere
        output_array = np.array(output_resized).astype(np.float32)
        base_array = np.array(base_img).astype(np.float32)
        mask_array = np.array(mask_resized).astype(np.float32) / 255.0
        
        # Expand mask to 3 channels (RGB)
        mask_3d = np.stack([mask_array] * 3, axis=2)
        
        # Blend: use model output in masked areas, original in unmasked areas
        # Smooth transition at mask edges
        blended = base_array * (1.0 - mask_3d) + output_array * mask_3d
        
        # Convert back to PIL Image
        blended_img = Image.fromarray(blended.astype(np.uint8))
        
        # Add realistic surgical scar overlay to the regenerated area
        final_output = add_surgical_scar(blended_img, mask_resized)
        
        output_encoded = image_to_base64(final_output)
        # Also return original image for before/after comparison
        original_encoded = image_to_base64(base_img)
        return JSONResponse({"image": output_encoded, "original": original_encoded})
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


