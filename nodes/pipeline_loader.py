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


def get_vae_list():
    """Get list of available VAE models, including subdirectories."""
    _ensure_imports()
    vae_list = ["none"]  # Option to not load a separate VAE
    
    # Get VAEs from ComfyUI's vae folder(s)
    vae_dirs = folder_paths.get_folder_paths("vae")
    for vae_folder in vae_dirs:
        if os.path.exists(vae_folder):
            # Walk through all subdirectories
            for root, dirs, files in os.walk(vae_folder):
                for f in files:
                    if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
                        # Get relative path from vae folder
                        rel_path = os.path.relpath(os.path.join(root, f), vae_folder)
                        vae_list.append(rel_path)
    
    return vae_list


def get_checkpoint_list():
    """Get list of available checkpoint/diffusers models."""
    _ensure_imports()
    checkpoint_list = []
    
    # Add recommended HuggingFace models at the top
    hf_models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
    ]
    checkpoint_list.extend(hf_models)
    
    # Get checkpoints from ComfyUI's checkpoints folder(s)
    try:
        ckpt_dirs = folder_paths.get_folder_paths("checkpoints")
        for ckpt_folder in ckpt_dirs:
            if os.path.exists(ckpt_folder):
                for root, dirs, files in os.walk(ckpt_folder):
                    for f in files:
                        if f.endswith(('.safetensors', '.ckpt')):
                            # Get relative path from checkpoint folder
                            rel_path = os.path.relpath(os.path.join(root, f), ckpt_folder)
                            # Only include SDXL-like models (heuristic: larger files or 'xl' in name)
                            checkpoint_list.append(rel_path)
    except Exception:
        pass
    
    # Get diffusers models from ComfyUI's diffusers folder if it exists
    try:
        diffusers_dirs = folder_paths.get_folder_paths("diffusers")
        for diff_folder in diffusers_dirs:
            if os.path.exists(diff_folder):
                for item in os.listdir(diff_folder):
                    item_path = os.path.join(diff_folder, item)
                    # Check if it's a diffusers model directory (has model_index.json)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "model_index.json")):
                        checkpoint_list.append(f"diffusers:{item}")
    except Exception:
        pass
    
    return checkpoint_list if checkpoint_list else ["stabilityai/stable-diffusion-xl-base-1.0"]


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
    
    def __init__(self):
        _ensure_imports()
        self.device = get_torch_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        _ensure_imports()
        return {
            "required": {
                "model_name": (get_checkpoint_list(), {
                    "default": "stabilityai/stable-diffusion-xl-base-1.0",
                    "tooltip": "Select a model from checkpoints folder or HuggingFace. For custom HF models, use model_path_override.",
                }),
                "model_type": (["SDXL", "SD2.1"], {
                    "default": "SDXL",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override model_name with a custom HuggingFace ID (e.g. 'SG161222/RealVisXL_V4.0'). Leave empty to use model_name.",
                }),
                "vae_name": (get_vae_list(), {
                    "default": "none",
                    "tooltip": "Select a VAE from models/vae folder, or use 'none' to use model's built-in VAE.",
                }),
                "vae_id": ("STRING", {
                    "default": "madebyollin/sdxl-vae-fp16-fix",
                    "multiline": False,
                    "tooltip": "HuggingFace VAE model ID. Recommended: madebyollin/sdxl-vae-fp16-fix. Leave empty to use vae_name instead.",
                }),
            }
        }
    
    RETURN_TYPES = ("MVADAPTER_PIPELINE", "VAE")
    RETURN_NAMES = ("pipeline", "vae")
    FUNCTION = "load_pipeline"
    CATEGORY = "MV-Adapter"
    
    def load_pipeline(
        self,
        model_name: str,
        model_type: str,
        auto_download: bool = True,
        model_path_override: str = "",
        vae_name: str = "none",
        vae_id: str = "",
    ) -> Tuple[Any, Any]:
        """Load the MV-Adapter pipeline."""
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file
        
        # Import our bundled MV-Adapter pipeline
        from ..mvadapter.pipelines import MVAdapterI2MVSDXLPipeline
        
        _ensure_imports()
        
        # Determine model path: override takes priority, then model_name
        if model_path_override and model_path_override.strip():
            model_path = model_path_override.strip()
        else:
            model_path = model_name
        
        # Handle diffusers: prefix for local diffusers models
        if model_path.startswith("diffusers:"):
            diffusers_name = model_path[len("diffusers:"):]
            diffusers_dirs = folder_paths.get_folder_paths("diffusers")
            for diff_folder in diffusers_dirs:
                potential_path = os.path.join(diff_folder, diffusers_name)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
        
        # Handle local checkpoint files
        elif not "/" in model_path or not model_path.startswith(("stabilityai/", "huggingface/", "Lykon/", "cagliostrolab/")):
            # Check if it's a local checkpoint file
            ckpt_dirs = folder_paths.get_folder_paths("checkpoints")
            for ckpt_folder in ckpt_dirs:
                potential_path = os.path.join(ckpt_folder, model_path)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
        
        # Auto-detect best dtype for the GPU
        dtype = torch.float32  # default fallback
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            
            # Check if it's AMD/ROCm
            is_amd = "amd" in device_name or "radeon" in device_name or "gfx" in device_name
            
            if is_amd:
                # AMD GPUs: Test bfloat16 support directly
                try:
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
                    _ = test_tensor * 2  # Simple operation to verify it works
                    dtype = torch.bfloat16
                    print(f"[MV-Adapter] AMD GPU detected ({device_name}), using bfloat16")
                except Exception:
                    dtype = torch.float16
                    print(f"[MV-Adapter] AMD GPU detected ({device_name}), bfloat16 not supported, using float16")
            else:
                # NVIDIA GPUs: Check compute capability
                try:
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 8:  # Ampere (RTX 30xx, A100) or newer
                        dtype = torch.bfloat16
                        print(f"[MV-Adapter] NVIDIA GPU (compute {capability[0]}.{capability[1]}), using bfloat16")
                    else:
                        dtype = torch.float16
                        print(f"[MV-Adapter] NVIDIA GPU (compute {capability[0]}.{capability[1]}), using float16")
                except Exception:
                    dtype = torch.float16
                    print(f"[MV-Adapter] Could not detect GPU capability, using float16")
        else:
            print(f"[MV-Adapter] No CUDA GPU, using float32")
        
        # Check if model exists locally or needs download
        is_hf_model = "/" in model_path and not os.path.exists(model_path)
        
        if is_hf_model and not auto_download:
            raise ValueError(
                f"Model '{model_path}' not found locally and auto_download is disabled. "
                f"Enable auto_download or provide a local path."
            )
        
        if is_hf_model and auto_download:
            print(f"[MV-Adapter] Model will be downloaded from HuggingFace: {model_path}")
        
        # Load VAE - priority: vae_id (HuggingFace) > vae_name (local file)
        vae = None
        
        # First try HuggingFace VAE ID if provided
        if vae_id and vae_id.strip() and "/" in vae_id:
            print(f"[MV-Adapter] Loading VAE from HuggingFace: {vae_id}")
            try:
                vae = AutoencoderKL.from_pretrained(
                    vae_id.strip(),
                    torch_dtype=dtype,
                )
            except Exception as e:
                print(f"[MV-Adapter] Failed to load VAE from HuggingFace: {e}")
                vae = None
        
        # If no HF VAE, try local file
        if vae is None and vae_name and vae_name != "none":
            # Find VAE file in ComfyUI's vae folder
            vae_path = None
            for vae_folder in folder_paths.get_folder_paths("vae"):
                potential_path = os.path.join(vae_folder, vae_name)
                if os.path.exists(potential_path):
                    vae_path = potential_path
                    break
            
            if vae_path:
                print(f"[MV-Adapter] Loading VAE from {vae_path}")
                vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=dtype,
                )
            else:
                print(f"[MV-Adapter] Warning: VAE '{vae_name}' not found in vae folders")
        
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
        }
        
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        
        try:
            # Check if it's a single file checkpoint (.safetensors or .ckpt)
            is_single_file = model_path.endswith(('.safetensors', '.ckpt'))
            
            if is_single_file:
                print(f"[MV-Adapter] Loading from single file checkpoint")
                pipeline = pipeline_cls.from_single_file(
                    model_path,
                    **pipeline_kwargs,
                )
            else:
                # Load from HuggingFace or diffusers directory
                pipeline_kwargs["use_safetensors"] = True
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
        
        # Enable memory optimizations (VAE only - attention slicing breaks MV-Adapter)
        pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        # NOTE: Do NOT enable_attention_slicing - it replaces MV-Adapter's custom attention processors
        
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
