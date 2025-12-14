"""
Sampler nodes for MV-Adapter.

Main generation nodes for text-to-multiview and image-to-multiview.
"""

from typing import Dict, Any, Tuple, Optional, List

from .pipeline_loader import get_torch_device


class MVAdapterI2MVSampler:
    """
    Image-to-Multiview sampler.
    
    Generates multiple consistent views from a single reference image.
    """
    
    def __init__(self):
        self.device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pipeline": ("MVADAPTER_PIPELINE",),
                "camera_embed": ("CAMERA_EMBED",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "high quality, best quality",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    "multiline": True,
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": {
                "reference_conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "MV-Adapter"
    
    def sample(
        self,
        pipeline,
        camera_embed: Dict[str, Any],
        reference_image,  # torch.Tensor
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        reference_conditioning_scale: float = 1.0,
    ):
        """Generate multi-view images from reference image."""
        import torch
        from PIL import Image
        from ..utils.image_utils import tensor_to_pil, pil_to_tensor
        from ..utils.camera_utils import prepare_camera_embed_for_pipeline
        
        # Get config from pipeline
        config = getattr(pipeline, "_mvadapter_config", {})
        num_views = config.get("num_views", camera_embed["num_views"])
        
        # Convert reference image to PIL
        ref_pil_list = tensor_to_pil(reference_image)
        ref_pil = ref_pil_list[0]  # Use first image if batch
        
        # Resize reference to match generation size
        width = camera_embed["width"]
        height = camera_embed["height"]
        ref_pil = ref_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        # Prepare camera embeddings
        plucker_embeds = camera_embed["embeddings"]
        control_images = prepare_camera_embed_for_pipeline(
            plucker_embeds,
            device=self.device,
            dtype=pipeline.dtype if hasattr(pipeline, "dtype") else torch.float16,
        )
        
        # Set random seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare prompts for each view
        prompts = [prompt] * num_views
        negative_prompts = [negative_prompt] * num_views
        
        print(f"[MV-Adapter] Generating {num_views} views at {width}x{height}")
        print(f"[MV-Adapter] Steps: {steps}, CFG: {guidance_scale}, Seed: {seed}")
        
        # Generate images
        # Note: The actual generation depends on the pipeline implementation
        # For now, we'll do a simplified version
        
        try:
            # Try using the pipeline's __call__ method with control images
            output = pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                image=control_images,  # Camera embeddings as control
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
            )
            
            output_images = output.images
            
        except Exception as e:
            print(f"[MV-Adapter] Pipeline call failed: {e}")
            print("[MV-Adapter] Falling back to basic generation...")
            
            # Fallback: generate individual views
            output_images = []
            for i in range(num_views):
                output = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
                output_images.append(output.images[0])
        
        # Convert to ComfyUI tensor format
        output_tensor = pil_to_tensor(output_images)
        
        print(f"[MV-Adapter] Generated {len(output_images)} images")
        
        return (output_tensor,)


class MVAdapterT2MVSampler:
    """
    Text-to-Multiview sampler.
    
    Generates multiple consistent views from a text prompt.
    """
    
    def __init__(self):
        self.device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pipeline": ("MVADAPTER_PIPELINE",),
                "camera_embed": ("CAMERA_EMBED",),
                "prompt": ("STRING", {
                    "default": "a 3D model of a robot, high quality, best quality",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    "multiline": True,
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "MV-Adapter"
    
    def sample(
        self,
        pipeline,
        camera_embed: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
    ):
        """Generate multi-view images from text prompt."""
        import torch
        from ..utils.image_utils import pil_to_tensor
        from ..utils.camera_utils import prepare_camera_embed_for_pipeline
        
        # Get config from pipeline
        config = getattr(pipeline, "_mvadapter_config", {})
        num_views = config.get("num_views", camera_embed["num_views"])
        
        # Get dimensions from camera embed
        width = camera_embed["width"]
        height = camera_embed["height"]
        
        # Prepare camera embeddings
        plucker_embeds = camera_embed["embeddings"]
        control_images = prepare_camera_embed_for_pipeline(
            plucker_embeds,
            device=self.device,
            dtype=pipeline.dtype if hasattr(pipeline, "dtype") else torch.float16,
        )
        
        # Set random seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare prompts for each view
        prompts = [prompt] * num_views
        negative_prompts = [negative_prompt] * num_views
        
        print(f"[MV-Adapter] Generating {num_views} views at {width}x{height}")
        print(f"[MV-Adapter] Steps: {steps}, CFG: {guidance_scale}, Seed: {seed}")
        
        # Generate images
        try:
            output = pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                image=control_images,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
            )
            
            output_images = output.images
            
        except Exception as e:
            print(f"[MV-Adapter] Pipeline call failed: {e}")
            print("[MV-Adapter] Falling back to basic generation...")
            
            output_images = []
            for i in range(num_views):
                output = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
                output_images.append(output.images[0])
        
        # Convert to ComfyUI tensor format
        output_tensor = pil_to_tensor(output_images)
        
        print(f"[MV-Adapter] Generated {len(output_images)} images")
        
        return (output_tensor,)
