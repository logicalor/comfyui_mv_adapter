"""
Pipeline loader nodes for MV-Adapter.

Loads diffusers pipelines for multi-view generation.
"""

import os
import torch
import folder_paths
from typing import Dict, Any, Tuple, Optional

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)


# Register custom model paths for MV-Adapter
MVADAPTER_MODELS_DIR = os.path.join(folder_paths.models_dir, "mvadapter")
os.makedirs(MVADAPTER_MODELS_DIR, exist_ok=True)


def get_torch_device():
    """Get the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MVAdapterPipelineLoader:
    """
    Load a diffusers pipeline for MV-Adapter multi-view generation.
    
    Supports SDXL and SD2.1 base models from HuggingFace or local paths.
    """
    
    def __init__(self):
        self.device = get_torch_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "stabilityai/stable-diffusion-xl-base-1.0",
                    "multiline": False,
                }),
                "model_type": (["SDXL", "SD2.1"], {
                    "default": "SDXL",
                }),
                "torch_dtype": (["float16", "float32", "bfloat16"], {
                    "default": "float16",
                }),
            },
            "optional": {
                "vae_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("MVADAPTER_PIPELINE", "VAE")
    RETURN_NAMES = ("pipeline", "vae")
    FUNCTION = "load_pipeline"
    CATEGORY = "MV-Adapter"
    
    def load_pipeline(
        self,
        model_path: str,
        model_type: str,
        torch_dtype: str,
        vae_path: str = "",
    ) -> Tuple[Any, Any]:
        """Load the diffusers pipeline."""
        
        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[torch_dtype]
        
        # Load VAE if specified
        vae = None
        if vae_path and vae_path.strip():
            print(f"[MV-Adapter] Loading VAE from {vae_path}")
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=dtype,
            )
        
        # Select pipeline class based on model type
        if model_type == "SDXL":
            pipeline_cls = StableDiffusionXLPipeline
        else:
            pipeline_cls = StableDiffusionPipeline
        
        # Load pipeline
        print(f"[MV-Adapter] Loading pipeline from {model_path}")
        
        pipeline_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        
        pipeline = pipeline_cls.from_pretrained(
            model_path,
            **pipeline_kwargs,
        )
        
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimizations
        pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        
        print(f"[MV-Adapter] Pipeline loaded successfully on {self.device}")
        
        return (pipeline, pipeline.vae)


class MVAdapterSchedulerConfig:
    """
    Configure the scheduler for MV-Adapter generation.
    
    Supports SNR shifting for improved multi-view quality.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pipeline": ("MVADAPTER_PIPELINE",),
                "scheduler_type": ([
                    "DDIM",
                    "Euler",
                    "EulerAncestral",
                ], {
                    "default": "Euler",
                }),
                "shift_snr": ("BOOLEAN", {
                    "default": True,
                }),
                "shift_scale": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
            },
        }
    
    RETURN_TYPES = ("MVADAPTER_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "configure_scheduler"
    CATEGORY = "MV-Adapter"
    
    def configure_scheduler(
        self,
        pipeline,
        scheduler_type: str,
        shift_snr: bool,
        shift_scale: float,
    ):
        """Configure the scheduler with optional SNR shifting."""
        from ..mvadapter.scheduler import create_scheduler_with_shift
        
        # Create base scheduler
        scheduler_config = pipeline.scheduler.config
        
        if scheduler_type == "DDIM":
            scheduler = DDIMScheduler.from_config(scheduler_config)
        elif scheduler_type == "Euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler_type == "EulerAncestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        else:
            scheduler = DDIMScheduler.from_config(scheduler_config)
        
        # Apply SNR shift if enabled
        if shift_snr:
            scheduler = create_scheduler_with_shift(
                scheduler=scheduler,
                shift_snr=True,
                shift_mode="interpolated",
                shift_scale=shift_scale,
            )
        
        pipeline.scheduler = scheduler
        
        print(f"[MV-Adapter] Scheduler configured: {scheduler_type}, SNR shift: {shift_snr}")
        
        return (pipeline,)
