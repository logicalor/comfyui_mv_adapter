"""
Utility nodes for MV-Adapter.

Background removal, image preprocessing, and grid utilities.
"""

from typing import Dict, Any, Tuple, Optional

from .pipeline_loader import get_torch_device


class MVAdapterBackgroundRemoval:
    """
    Remove background from images for better multi-view generation.
    
    Uses rembg for background removal and optional preprocessing.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "bg_color": (["gray", "white", "black", "transparent"], {
                    "default": "gray",
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_background"
    CATEGORY = "MV-Adapter"
    
    BG_COLORS = {
        "gray": (127, 127, 127),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "transparent": None,
    }
    
    def remove_background(
        self,
        image,  # torch.Tensor
        bg_color: str = "gray",
    ):
        """Remove background from image."""
        from PIL import Image
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor
        
        try:
            from rembg import remove
        except ImportError:
            print("[MV-Adapter] Warning: rembg not installed. Install with: pip install rembg")
            return (image,)
        
        # Convert to PIL
        pil_images = tensor_to_pil(image)
        
        processed_images = []
        for pil_img in pil_images:
            # Remove background (returns RGBA)
            result = remove(pil_img)
            
            # Apply background color
            color = self.BG_COLORS[bg_color]
            if color is not None:
                # Create background and composite
                background = Image.new("RGB", result.size, color)
                background.paste(result, mask=result.split()[3])
                result = background
            else:
                # Keep as RGBA, convert to RGB for ComfyUI
                result = result.convert("RGB")
            
            processed_images.append(result)
        
        # Convert back to tensor
        output_tensor = pil_to_tensor(processed_images)
        
        print(f"[MV-Adapter] Background removed from {len(pil_images)} image(s)")
        
        return (output_tensor,)


class MVAdapterImagePreprocess:
    """
    Preprocess images for MV-Adapter generation.
    
    Resizes and pads images to the correct size for generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
                "resize_mode": (["contain", "cover", "stretch"], {
                    "default": "contain",
                }),
            },
            "optional": {
                "bg_color": (["gray", "white", "black"], {
                    "default": "gray",
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preprocess"
    CATEGORY = "MV-Adapter"
    
    BG_COLORS = {
        "gray": (127, 127, 127),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    
    def preprocess(
        self,
        image,  # torch.Tensor
        target_size: int,
        resize_mode: str,
        bg_color: str = "gray",
    ):
        """Preprocess image for generation."""
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor, resize_image
        
        # Convert to PIL
        pil_images = tensor_to_pil(image)
        
        color = self.BG_COLORS[bg_color]
        
        processed_images = []
        for pil_img in pil_images:
            resized = resize_image(
                pil_img,
                target_size=target_size,
                mode=resize_mode,
                bg_color=color,
            )
            processed_images.append(resized)
        
        # Convert back to tensor
        output_tensor = pil_to_tensor(processed_images)
        
        print(f"[MV-Adapter] Preprocessed {len(pil_images)} image(s) to {target_size}x{target_size}")
        
        return (output_tensor,)


class MVAdapterImageGrid:
    """
    Combine multiple images into a grid.
    
    Useful for visualizing all views together.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "columns": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 12,
                    "step": 1,
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                }),
                "bg_color": (["white", "black", "gray"], {
                    "default": "white",
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "create_grid"
    CATEGORY = "MV-Adapter"
    
    BG_COLORS = {
        "gray": (127, 127, 127),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    
    def create_grid(
        self,
        images,  # torch.Tensor
        columns: int = 0,
        padding: int = 0,
        bg_color: str = "white",
    ):
        """Create image grid from batch."""
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor, create_image_grid
        
        # Convert to PIL
        pil_images = tensor_to_pil(images)
        
        if len(pil_images) == 0:
            return (images,)
        
        # Auto-calculate columns if not specified
        cols = columns if columns > 0 else None
        
        color = self.BG_COLORS[bg_color]
        
        # Create grid
        grid = create_image_grid(
            pil_images,
            cols=cols,
            padding=padding,
            bg_color=color,
        )
        
        # Convert back to tensor
        output_tensor = pil_to_tensor([grid])
        
        print(f"[MV-Adapter] Created grid from {len(pil_images)} images")
        
        return (output_tensor,)


class MVAdapterSplitViews:
    """
    Split a batch of images into individual outputs.
    
    Useful for processing individual views separately.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("view_0", "view_1", "view_2", "view_3", "view_4", "view_5")
    FUNCTION = "split_views"
    CATEGORY = "MV-Adapter"
    
    def split_views(
        self,
        images,  # torch.Tensor
    ):
        """Split batch into individual views."""
        import torch
        
        batch_size = images.shape[0]
        
        # Pad with zeros if fewer than 6 images
        outputs = []
        for i in range(6):
            if i < batch_size:
                outputs.append(images[i:i+1])
            else:
                # Return empty tensor with same HWC dimensions
                h, w, c = images.shape[1:]
                outputs.append(torch.zeros(1, h, w, c))
        
        return tuple(outputs)


class MVAdapterClearVRAM:
    """
    Clear VRAM by forcing garbage collection and emptying CUDA cache.
    
    Place this node between heavy operations to free up memory.
    The node passes through the LATENT input unchanged but clears memory first.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latents": ("LATENT",),
            },
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "clear_vram"
    CATEGORY = "MV-Adapter"
    
    def clear_vram(self, latents):
        """Clear VRAM and pass through latents."""
        import gc
        import torch
        
        # Validate input
        if latents is None:
            raise ValueError("[MV-Adapter Clear VRAM] latents input is None. Make sure output_type is set to 'latents' in the sampler and you're connecting the 'latents' output (not 'images').")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[MV-Adapter] VRAM cleared. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            print("[MV-Adapter] VRAM cleared (no CUDA device)")
        
        return (latents,)


class MVAdapterVAEDecode:
    """
    Memory-efficient VAE decode for MV-Adapter latents.
    
    Decodes latents one at a time with aggressive memory cleanup
    between each decode. Uses tiled VAE decoding for large images.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latents": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MV-Adapter"
    
    def decode(
        self,
        latents: Dict[str, Any],
        vae,
        tile_size: int = 512,
    ):
        """Decode latents with memory efficiency."""
        import gc
        import torch
        
        # Validate input
        if latents is None:
            raise ValueError("[MV-Adapter VAE Decode] latents input is None. Make sure output_type is set to 'latents' in the sampler and you're connecting the 'latents' output (not 'images').")
        
        if not isinstance(latents, dict):
            raise ValueError(f"[MV-Adapter VAE Decode] Expected latents to be a dict, got {type(latents)}")
        
        if "samples" not in latents:
            raise ValueError(f"[MV-Adapter VAE Decode] latents dict missing 'samples' key. Keys present: {list(latents.keys())}")
        
        samples = latents["samples"]
        
        if samples is None:
            raise ValueError("[MV-Adapter VAE Decode] latents['samples'] is None")
        batch_size = samples.shape[0]
        
        print(f"[MV-Adapter] Memory-efficient VAE decode: {batch_size} images")
        
        # Get device
        device = get_torch_device()
        
        # Clear VRAM before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        decoded_images = []
        
        # Decode one at a time for memory efficiency
        for i in range(batch_size):
            print(f"[MV-Adapter] Decoding image {i+1}/{batch_size}...")
            
            # Get single latent
            single_latent = samples[i:i+1].to(device)
            
            # Use ComfyUI's VAE decode
            decoded = vae.decode(single_latent)
            
            # Move to CPU immediately to free VRAM
            decoded_images.append(decoded.cpu())
            
            # Clear intermediate tensors
            del single_latent, decoded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate results
        output = torch.cat(decoded_images, dim=0)
        
        # Final cleanup
        del decoded_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[MV-Adapter] VAE decode complete: {output.shape}")
        
        return (output,)
