"""
Pipeline loader nodes for MV-Adapter.

Loads diffusers pipelines for multi-view generation.
"""

import os
from typing import Dict, Any, Tuple, Optional

# Defer heavy imports to runtime
torch = None
folder_paths = None

def _ensure_imports():
    """Lazy import heavy dependencies."""
    global torch, folder_paths
    if torch is None:
        import torch as _torch
        torch = _torch
    if folder_paths is None:
        import folder_paths as _folder_paths
        folder_paths = _folder_paths


def get_mvadapter_models_dir():
    """Get MV-Adapter models directory."""
    _ensure_imports()
    models_dir = os.path.join(folder_paths.models_dir, "mvadapter")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_torch_device():
    """Get the best available torch device."""
    _ensure_imports()
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
    
    # Common base models for multi-view generation
    RECOMMENDED_MODELS = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "Lykon/dreamshaper-xl-1-0",
        "cagliostrolab/animagine-xl-3.1",
    ]
    
    def __init__(self):
        _ensure_imports()
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
                "auto_download": ("BOOLEAN", {
                    "default": True,
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
        auto_download: bool = True,
        vae_path: str = "",
    ) -> Tuple[Any, Any]:
        """Load the MV-Adapter pipeline."""
        from diffusers import AutoencoderKL
        
        # Import our bundled MV-Adapter pipeline
        from ..mvadapter.pipelines import MVAdapterI2MVSDXLPipeline
        
        _ensure_imports()
        
        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[torch_dtype]
        
        # Check if model exists locally or needs download
        is_hf_model = "/" in model_path and not os.path.exists(model_path)
        
        if is_hf_model and not auto_download:
            raise ValueError(
                f"Model '{model_path}' not found locally and auto_download is disabled. "
                f"Enable auto_download or provide a local path."
            )
        
        if is_hf_model and auto_download:
            print(f"[MV-Adapter] Model will be downloaded from HuggingFace: {model_path}")
        
        # Load VAE if specified
        vae = None
        if vae_path and vae_path.strip():
            print(f"[MV-Adapter] Loading VAE from {vae_path}")
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=dtype,
            )
        
        # Select pipeline class based on model type
        # Currently only SDXL is supported with bundled pipeline
        if model_type == "SDXL":
            pipeline_cls = MVAdapterI2MVSDXLPipeline
        else:
            # For SD2.1, fall back to standard pipeline (TODO: bundle SD2.1 pipeline)
            from diffusers import StableDiffusionPipeline
            pipeline_cls = StableDiffusionPipeline
            print(f"[MV-Adapter] Warning: SD2.1 uses standard pipeline without MV-Adapter features")
        
        # Load pipeline
        print(f"[MV-Adapter] Loading MV-Adapter pipeline from {model_path}")
        
        pipeline_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        
        try:
            pipeline = pipeline_cls.from_pretrained(
                model_path,
                **pipeline_kwargs,
            )
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise ValueError(
                    f"Model '{model_path}' not found. Check the model path or HuggingFace repo name."
                ) from e
            raise
        
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimizations
        pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("auto")
        
        # Clear VRAM after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        from diffusers import (
            DDIMScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
        )
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
