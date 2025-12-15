"""
Utility nodes for MV-Adapter.

Background removal, image preprocessing, and grid utilities.
"""

from typing import Dict, Any, Tuple, Optional

from .pipeline_loader import get_torch_device


class MVAdapterBackgroundRemoval:
    """
    Remove background from images for better multi-view generation.
    
    Supports two methods:
    - BiRefNet: Higher quality, better for hair and fine details (recommended)
    - rembg: Faster, good general purpose
    """
    
    _birefnet_model = None
    _birefnet_transform = None
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "method": (["birefnet", "rembg"], {
                    "default": "birefnet",
                    "tooltip": "BiRefNet is higher quality (used by official demo). rembg is faster.",
                }),
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
    
    @classmethod
    def _load_birefnet(cls):
        """Load BiRefNet model (cached)."""
        if cls._birefnet_model is None:
            import torch
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation
            
            print("[MV-Adapter] Loading BiRefNet model...")
            cls._birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            
            device = get_torch_device()
            cls._birefnet_model.to(device)
            cls._birefnet_model.eval()
            
            cls._birefnet_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            print("[MV-Adapter] BiRefNet model loaded")
        
        return cls._birefnet_model, cls._birefnet_transform
    
    def _remove_bg_birefnet(self, pil_img):
        """Remove background using BiRefNet."""
        import torch
        from PIL import Image
        
        model, transform = self._load_birefnet()
        device = get_torch_device()
        
        # Store original size
        original_size = pil_img.size
        
        # Transform image
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        
        # Get mask
        pred = preds[0].squeeze()
        mask = (pred * 255).byte().numpy()
        
        # Resize mask to original size
        mask_img = Image.fromarray(mask).resize(original_size, Image.BILINEAR)
        
        # Apply mask to original image
        pil_img = pil_img.convert("RGBA")
        pil_img.putalpha(mask_img)
        
        return pil_img
    
    def _remove_bg_rembg(self, pil_img):
        """Remove background using rembg."""
        from rembg import remove
        return remove(pil_img)
    
    def remove_background(
        self,
        image,  # torch.Tensor
        method: str = "birefnet",
        bg_color: str = "gray",
    ):
        """Remove background from image."""
        from PIL import Image
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor
        
        # Check dependencies
        if method == "rembg":
            try:
                from rembg import remove
            except ImportError:
                print("[MV-Adapter] Warning: rembg not installed. Install with: pip install rembg")
                print("[MV-Adapter] Falling back to BiRefNet")
                method = "birefnet"
        
        if method == "birefnet":
            try:
                from transformers import AutoModelForImageSegmentation
            except ImportError:
                print("[MV-Adapter] Warning: transformers not installed for BiRefNet")
                return (image,)
        
        # Convert to PIL
        pil_images = tensor_to_pil(image)
        
        processed_images = []
        for i, pil_img in enumerate(pil_images):
            print(f"[MV-Adapter] Removing background from image {i+1}/{len(pil_images)} using {method}...")
            
            # Remove background (returns RGBA)
            if method == "birefnet":
                result = self._remove_bg_birefnet(pil_img)
            else:
                result = self._remove_bg_rembg(pil_img)
            
            # Apply background color
            color = self.BG_COLORS[bg_color]
            if color is not None:
                # Create background and composite
                background = Image.new("RGB", result.size, color)
                if result.mode == "RGBA":
                    background.paste(result, mask=result.split()[3])
                else:
                    background.paste(result)
                result = background
            else:
                # Keep as RGBA, convert to RGB for ComfyUI
                result = result.convert("RGB")
            
            processed_images.append(result)
        
        # Convert back to tensor
        output_tensor = pil_to_tensor(processed_images)
        
        print(f"[MV-Adapter] Background removed from {len(pil_images)} image(s) using {method}")
        
        return (output_tensor,)


class MVAdapterReferencePreprocess:
    """
    Official reference image preprocessing for MV-Adapter.
    
    Matches the exact preprocessing used in the official HuggingFace demo:
    - Crops to object bounding box (using alpha)
    - Resizes to 90% of target size
    - Centers on gray background
    
    Use this node AFTER background removal for best results.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Output resolution (768 recommended for SDXL)",
                }),
                "object_scale": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of the frame the object should fill (0.9 = 90%)",
                }),
                "bg_color": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Background color (0=black, 0.5=gray, 1=white). Gray (0.5) matches official demo.",
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preprocess"
    CATEGORY = "MV-Adapter"
    
    def preprocess(
        self,
        image,  # torch.Tensor BHWC
        target_size: int = 768,
        object_scale: float = 0.9,
        bg_color: float = 0.5,
    ):
        """
        Preprocess reference image using official demo method.
        
        This matches the preprocess_image() function from:
        https://huggingface.co/spaces/VAST-AI/MV-Adapter-I2MV-SDXL/blob/main/inference_i2mv_sdxl.py
        """
        import numpy as np
        from PIL import Image
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor
        
        height = width = target_size
        
        # Convert to PIL
        pil_images = tensor_to_pil(image)
        
        processed_images = []
        for pil_img in pil_images:
            # Convert to RGBA if not already
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
            
            img_array = np.array(pil_img)
            alpha = img_array[..., 3] > 0
            H, W = alpha.shape
            
            # Get bounding box of alpha (object)
            y, x = np.where(alpha)
            
            if len(y) == 0 or len(x) == 0:
                # No alpha content - just resize
                result = np.array(pil_img.resize((width, height)))
                result = result[..., :3]  # Drop alpha
            else:
                # Crop to bounding box with 1px margin
                y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
                x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
                image_center = img_array[y0:y1, x0:x1]
                
                # Resize longer side to (target_size * object_scale)
                crop_H, crop_W, _ = image_center.shape
                target_dim = int(target_size * object_scale)
                
                if crop_H > crop_W:
                    new_W = int(crop_W * target_dim / crop_H)
                    new_H = target_dim
                else:
                    new_H = int(crop_H * target_dim / crop_W)
                    new_W = target_dim
                
                # Resize the cropped object
                image_center = np.array(
                    Image.fromarray(image_center).resize((new_W, new_H), Image.LANCZOS)
                )
                
                # Create output canvas and center the object
                start_h = (height - new_H) // 2
                start_w = (width - new_W) // 2
                result = np.zeros((height, width, 4), dtype=np.uint8)
                result[start_h:start_h + new_H, start_w:start_w + new_W] = image_center
                
                # Composite onto background color
                result = result.astype(np.float32) / 255.0
                result = result[:, :, :3] * result[:, :, 3:4] + (1 - result[:, :, 3:4]) * bg_color
                result = (result * 255).clip(0, 255).astype(np.uint8)
            
            processed_images.append(Image.fromarray(result))
        
        # Convert back to tensor
        output_tensor = pil_to_tensor(processed_images)
        
        print(f"[MV-Adapter] Reference preprocessed: {len(pil_images)} image(s) to {target_size}x{target_size}")
        
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
        
        # Debug: print latent stats
        print(f"[MV-Adapter] Latent stats before scaling - min: {samples.min().item():.4f}, max: {samples.max().item():.4f}, mean: {samples.mean().item():.4f}")
        
        # Scale latents for VAE decode
        # The pipeline outputs raw latents when output_type="latent", but VAE expects scaled latents
        # SDXL VAE scaling_factor is 0.13025 - we need to divide by it before decode
        scaling_factor = 0.13025
        samples = samples / scaling_factor
        print(f"[MV-Adapter] Latent stats after scaling (1/{scaling_factor}) - min: {samples.min().item():.4f}, max: {samples.max().item():.4f}")
        
        batch_size = samples.shape[0]
        
        print(f"[MV-Adapter] Memory-efficient VAE decode: {batch_size} images")
        
        # Get device
        device = get_torch_device()
        
        # Clear VRAM before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine VAE dtype from model weights
        vae_dtype = torch.float32  # default fallback
        if hasattr(vae, 'post_quant_conv') and hasattr(vae.post_quant_conv, 'weight'):
            vae_dtype = vae.post_quant_conv.weight.dtype
        elif hasattr(vae, 'dtype'):
            vae_dtype = vae.dtype
        
        print(f"[MV-Adapter] Detected VAE dtype: {vae_dtype}")
        
        # Only float16 needs upcasting to float32 (can cause NaN due to limited dynamic range)
        # bfloat16 and float32 are fine as-is
        needs_upcasting = (vae_dtype == torch.float16)
        original_dtype = vae_dtype
        
        if needs_upcasting:
            print(f"[MV-Adapter] Upcasting VAE from float16 to float32 for stable decode")
            vae.to(dtype=torch.float32)
            decode_dtype = torch.float32
        else:
            decode_dtype = vae_dtype
        
        print(f"[MV-Adapter] Using decode dtype: {decode_dtype}")
        
        decoded_images = []
        
        # Decode one at a time for memory efficiency
        for i in range(batch_size):
            print(f"[MV-Adapter] Decoding image {i+1}/{batch_size}...")
            
            # Get single latent - convert to VAE's dtype
            single_latent = samples[i:i+1].to(device=device, dtype=decode_dtype)
            
            # Use VAE decode
            decoded = vae.decode(single_latent)
            
            # Handle different return types (tensor, DecoderOutput, tuple)
            if hasattr(decoded, 'sample'):
                # DecoderOutput object
                decoded_tensor = decoded.sample
            elif isinstance(decoded, tuple):
                decoded_tensor = decoded[0]
            else:
                decoded_tensor = decoded
            
            print(f"[MV-Adapter] Decoded tensor {i+1} - shape: {decoded_tensor.shape}, min: {decoded_tensor.min().item():.4f}, max: {decoded_tensor.max().item():.4f}")
            
            # Move to CPU immediately to free VRAM
            decoded_images.append(decoded_tensor.float().cpu())
            
            # Clear intermediate tensors
            del single_latent, decoded, decoded_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore VAE dtype if we upcasted
        if needs_upcasting:
            print(f"[MV-Adapter] Restoring VAE to {original_dtype}")
            vae.to(dtype=original_dtype)
        
        # Concatenate results
        output = torch.cat(decoded_images, dim=0)
        
        print(f"[MV-Adapter] Raw decode output - shape: {output.shape}, min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        
        # Convert from NCHW (VAE output) to NHWC (ComfyUI format)
        # VAE outputs: [B, C, H, W] -> ComfyUI expects: [B, H, W, C]
        if output.shape[1] == 3:  # Channels in position 1 (NCHW)
            output = output.permute(0, 2, 3, 1)
        
        # Normalize based on actual range
        out_min = output.min().item()
        out_max = output.max().item()
        
        if out_min < -0.5:
            # Output is in approximately -1 to 1 range (diffusers style)
            print(f"[MV-Adapter] Normalizing from [-1,1] to [0,1]")
            output = (output + 1.0) / 2.0
        elif out_max > 1.5:
            # Output might be in 0-255 range
            print(f"[MV-Adapter] Normalizing from [0,255] to [0,1]")
            output = output / 255.0
        else:
            # Output is likely already in 0-1 range
            print(f"[MV-Adapter] Output appears to be in [0,1] range, no normalization needed")
        
        # Clamp to valid range
        output = output.clamp(0, 1)
        
        print(f"[MV-Adapter] After normalization - min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        
        # Final cleanup
        del decoded_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[MV-Adapter] VAE decode complete: {output.shape}")
        
        return (output,)
