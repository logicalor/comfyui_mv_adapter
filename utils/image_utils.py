"""
Image utilities for MV-Adapter.
Handles image format conversions and preprocessing.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional


def pil_to_tensor(images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    Convert PIL image(s) to ComfyUI tensor format.
    
    ComfyUI uses BHWC format with values in [0, 1].
    
    Args:
        images: Single PIL image or list of PIL images
        
    Returns:
        Tensor in BHWC format [B, H, W, C]
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    tensors = []
    for img in images:
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy array and normalize to [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0
        tensors.append(arr)
    
    # Stack into batch
    batch = np.stack(tensors, axis=0)
    return torch.from_numpy(batch)


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert ComfyUI tensor to PIL images.
    
    Args:
        tensor: Tensor in BHWC format [B, H, W, C]
        
    Returns:
        List of PIL images
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    # Move to CPU and convert to numpy
    tensor = tensor.cpu().numpy()
    
    # Clip to [0, 1] and convert to uint8
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)
    
    images = []
    for i in range(tensor.shape[0]):
        img = Image.fromarray(tensor[i])
        images.append(img)
    
    return images


def resize_image(
    image: Image.Image,
    target_size: int,
    mode: str = "contain",
    bg_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: Input PIL image
        target_size: Target size (square)
        mode: "contain" (fit within, pad), "cover" (fill, crop), "stretch"
        bg_color: Background color for padding
        
    Returns:
        Resized PIL image
    """
    if mode == "stretch":
        return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    w, h = image.size
    aspect = w / h
    
    if mode == "contain":
        if aspect > 1:
            new_w = target_size
            new_h = int(target_size / aspect)
        else:
            new_h = target_size
            new_w = int(target_size * aspect)
        
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create background and paste centered
        result = Image.new("RGB", (target_size, target_size), bg_color)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        
        if resized.mode == "RGBA":
            result.paste(resized, (paste_x, paste_y), resized)
        else:
            result.paste(resized, (paste_x, paste_y))
        
        return result
    
    elif mode == "cover":
        if aspect > 1:
            new_h = target_size
            new_w = int(target_size * aspect)
        else:
            new_w = target_size
            new_h = int(target_size / aspect)
        
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return resized.crop((left, top, right, bottom))
    
    return image


def preprocess_reference_image(
    image: Image.Image,
    target_size: int = 768,
    remove_bg: bool = True,
    bg_color: tuple = (127, 127, 127),
) -> Image.Image:
    """
    Preprocess a reference image for MV-Adapter.
    
    Args:
        image: Input PIL image
        target_size: Target size (768 for SDXL, 512 for SD2.1)
        remove_bg: Whether to remove background
        bg_color: Background color to use
        
    Returns:
        Preprocessed PIL image
    """
    if remove_bg:
        try:
            from rembg import remove
            image = remove(image)
        except ImportError:
            print("Warning: rembg not installed, skipping background removal")
    
    # Resize with padding
    image = resize_image(image, target_size, mode="contain", bg_color=bg_color)
    
    return image


def create_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 0,
    bg_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL images (should all be same size)
        rows: Number of rows (auto-calculated if None)
        cols: Number of columns (auto-calculated if None)
        padding: Padding between images
        bg_color: Background color
        
    Returns:
        Grid image as PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    n = len(images)
    
    # Auto-calculate grid dimensions
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    
    # Get image dimensions from first image
    w, h = images[0].size
    
    # Calculate grid size
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    
    # Create grid
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * (w + padding)
        y = row * (h + padding)
        
        if img.mode == "RGBA":
            grid.paste(img, (x, y), img)
        else:
            grid.paste(img, (x, y))
    
    return grid


def split_image_grid(
    grid: Image.Image,
    rows: int,
    cols: int,
    padding: int = 0,
) -> List[Image.Image]:
    """
    Split a grid image into individual images.
    
    Args:
        grid: Grid PIL image
        rows: Number of rows
        cols: Number of columns
        padding: Padding between images
        
    Returns:
        List of PIL images
    """
    grid_w, grid_h = grid.size
    
    # Calculate individual image size
    w = (grid_w - (cols - 1) * padding) // cols
    h = (grid_h - (rows - 1) * padding) // rows
    
    images = []
    for row in range(rows):
        for col in range(cols):
            x = col * (w + padding)
            y = row * (h + padding)
            img = grid.crop((x, y, x + w, y + h))
            images.append(img)
    
    return images
