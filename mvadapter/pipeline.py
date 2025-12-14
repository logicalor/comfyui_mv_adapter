"""
MV-Adapter Pipeline utilities for multi-view generation.

This provides helper functions for calling the official MV-Adapter pipelines.
Based on: https://github.com/huanngzh/MV-Adapter
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union


def prepare_reference_image(
    image: Image.Image,
    height: int,
    width: int,
    background_color: float = 0.5,
) -> Image.Image:
    """
    Preprocess reference image for MV-Adapter.
    Centers the object and pads to target size.
    """
    image = np.array(image)
    
    # Handle RGBA images
    if len(image.shape) == 3 and image.shape[-1] == 4:
        alpha = image[..., 3] > 0
        H, W = alpha.shape
        
        # Get bounding box of alpha
        y, x = np.where(alpha)
        if len(y) == 0 or len(x) == 0:
            # No alpha content, use full image
            image = image[..., :3]
        else:
            y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
            x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
            image_center = image[y0:y1, x0:x1]
            
            # Resize the longer side to 90% of target
            H, W, _ = image_center.shape
            if H > W:
                new_W = int(W * (height * 0.9) / H)
                new_H = int(height * 0.9)
            else:
                new_H = int(H * (width * 0.9) / W)
                new_W = int(width * 0.9)
            
            image_center = np.array(Image.fromarray(image_center).resize((new_W, new_H)))
            
            # Pad to target size
            start_h = (height - new_H) // 2
            start_w = (width - new_W) // 2
            image = np.zeros((height, width, 4), dtype=np.uint8)
            image[start_h:start_h + new_H, start_w:start_w + new_W] = image_center
            
            # Composite onto background
            image = image.astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * background_color
            image = (image * 255).clip(0, 255).astype(np.uint8)
    elif len(image.shape) == 3 and image.shape[-1] == 3:
        # RGB image - just resize
        image = np.array(Image.fromarray(image).resize((width, height)))
    else:
        # Other format - convert to RGB and resize
        image = np.array(Image.fromarray(image).convert("RGB").resize((width, height)))
    
    return Image.fromarray(image)


def run_mvadapter_pipeline(
    pipeline,
    reference_image: Optional[Image.Image],
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]],
    control_images: torch.Tensor,
    num_views: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Optional[torch.Generator] = None,
    reference_conditioning_scale: float = 1.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
    low_vram_mode: bool = False,
    progress_callback: Optional[callable] = None,
    output_type: str = "pil",
) -> Union[List[Image.Image], torch.Tensor]:
    """
    Run the MV-Adapter pipeline for multi-view generation.
    
    This calls the pipeline's __call__ method with the correct parameters.
    Works with both I2MV (Image-to-Multi-View) and T2MV (Text-to-Multi-View) modes.
    """
    if device is None:
        device = pipeline.device if hasattr(pipeline, 'device') else torch.device("cuda")
    
    # Apply additional memory optimizations in low VRAM mode
    if low_vram_mode:
        print("[MV-Adapter Pipeline] Low VRAM mode - enabling memory optimizations")
        # Enable VAE tiling for large images
        if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'enable_tiling'):
            pipeline.vae.enable_tiling()
        # Enable VAE slicing to decode one image at a time
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
        # NOTE: Do NOT enable_attention_slicing - it replaces MV-Adapter's custom attention processors
        # and breaks reference image conditioning!
        # Clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Prepare reference image if provided
    ref_image = None
    if reference_image is not None:
        ref_image = prepare_reference_image(reference_image, height, width)
    else:
        # For T2MV mode, create a blank reference image
        # The pipeline still needs one but it won't be used much
        ref_image = Image.new("RGB", (width, height), color=(128, 128, 128))
    
    # Prepare control images (camera embeddings)
    # control_images should be [num_views, 6, H, W] in NCHW format
    control_images = control_images.to(device=device, dtype=dtype)
    
    # Ensure all pipeline components are on the correct device
    # Note: Skip if using CPU offload (meta tensors can't be moved directly)
    if hasattr(pipeline, 'cond_encoder') and pipeline.cond_encoder is not None:
        try:
            # Check if any parameter is on meta device
            is_meta = any(p.device.type == 'meta' for p in pipeline.cond_encoder.parameters())
            if not is_meta:
                pipeline.cond_encoder.to(device=device, dtype=dtype)
        except Exception as e:
            print(f"[MV-Adapter Pipeline] Note: Could not move cond_encoder to device: {e}")
    
    # Move attention processors to correct device (skip meta tensors)
    for name, processor in pipeline.unet.attn_processors.items():
        if hasattr(processor, 'to') and hasattr(processor, 'parameters'):
            try:
                # Check if any parameter is on meta device
                params = list(processor.parameters())
                if params and any(p.device.type == 'meta' for p in params):
                    continue  # Skip meta tensors - they'll be materialized by the offload hook
                processor.to(device=device, dtype=dtype)
            except Exception as e:
                # Silently skip if we can't move (e.g., CPU offload is managing this)
                pass
        elif hasattr(processor, 'to'):
            try:
                processor.to(device=device, dtype=dtype)
            except NotImplementedError:
                # Meta tensor - skip
                pass
    
    print(f"[MV-Adapter Pipeline] Running inference:")
    print(f"  - mode: {'I2MV' if reference_image is not None else 'T2MV'}")
    print(f"  - num_views (num_images_per_prompt): {num_views}")
    print(f"  - height x width: {height} x {width}")
    print(f"  - num_inference_steps: {num_inference_steps}")
    print(f"  - guidance_scale: {guidance_scale}")
    print(f"  - control_images shape: {control_images.shape}")
    if reference_image is not None:
        print(f"  - reference_image: {ref_image.size}")
        print(f"  - reference_conditioning_scale: {reference_conditioning_scale}")
    
    # Create step callback for progress reporting
    def step_callback(pipe, step_index, timestep, callback_kwargs):
        if progress_callback is not None:
            progress_callback(step_index + 1, num_inference_steps)
        return callback_kwargs
    
    # Build kwargs for the pipeline call
    # These match the MV-Adapter pipeline __call__ signature
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_images_per_prompt": num_views,
        # MV-Adapter specific parameters
        "control_image": control_images,
        "control_conditioning_scale": 1.0,
        "mv_scale": 1.0,
        "reference_image": ref_image,
        "reference_conditioning_scale": reference_conditioning_scale if reference_image is not None else 0.0,
        # Progress callback
        "callback_on_step_end": step_callback if progress_callback is not None else None,
        # Output type: "pil" for decoded images, "latent" for raw latents
        "output_type": output_type,
    }
    
    try:
        # Call the MV-Adapter pipeline
        output = pipeline(**kwargs)
        
        # Return based on output type
        if output_type == "latent":
            # Return latents directly
            if hasattr(output, 'latents'):
                return output.latents
            elif isinstance(output, tuple) and len(output) > 0:
                return output[0]  # Latents are first element
            else:
                return output
        else:
            # Return decoded images
            if hasattr(output, 'images'):
                return output.images
        else:
            return output[0]
            
    except Exception as e:
        print(f"[MV-Adapter Pipeline] Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Return placeholder images on error
        placeholder = Image.new("RGB", (width, height), color=(128, 128, 128))
        return [placeholder] * num_views
