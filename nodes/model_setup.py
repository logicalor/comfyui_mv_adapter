"""
Model setup node for MV-Adapter.

Initializes MV-Adapter attention processors and loads adapter weights.
"""

import os
from typing import Dict, Any, Tuple, List

from .pipeline_loader import get_mvadapter_models_dir, get_torch_device

# Defer heavy imports
torch = None
folder_paths = None

def _ensure_imports():
    global torch, folder_paths
    if torch is None:
        import torch as _torch
        torch = _torch
    if folder_paths is None:
        import folder_paths as _folder_paths
        folder_paths = _folder_paths


def get_mvadapter_models() -> List[str]:
    """Get list of available MV-Adapter model files."""
    models = []
    
    models_dir = get_mvadapter_models_dir()
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith((".safetensors", ".bin", ".pt")):
                models.append(f)
    
    # Add HuggingFace model options
    models.extend([
        "huanngzh/mv-adapter/mvadapter_i2mv_sdxl.safetensors",
        "huanngzh/mv-adapter/mvadapter_t2mv_sdxl.safetensors",
        "huanngzh/mv-adapter/mvadapter_i2mv_sd21.safetensors",
        "huanngzh/mv-adapter/mvadapter_t2mv_sd21.safetensors",
    ])
    
    return models if models else ["mvadapter_i2mv_sdxl.safetensors"]


class MVAdapterModelSetup:
    """
    Initialize MV-Adapter attention processors and load adapter weights.
    
    This node configures the UNet with multi-view attention and loads
    the adapter weights for consistent multi-view generation.
    """
    
    def __init__(self):
        self.device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pipeline": ("MVADAPTER_PIPELINE",),
                "adapter_path": ("STRING", {
                    "default": "huanngzh/mv-adapter/mvadapter_i2mv_sdxl.safetensors",
                    "multiline": False,
                }),
                "num_views": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 12,
                    "step": 1,
                }),
                "adapter_mode": (["i2mv", "t2mv"], {
                    "default": "i2mv",
                }),
            },
        }
    
    RETURN_TYPES = ("MVADAPTER_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "setup_adapter"
    CATEGORY = "MV-Adapter"
    
    def setup_adapter(
        self,
        pipeline,
        adapter_path: str,
        num_views: int,
        adapter_mode: str,
    ) -> Tuple[Any]:
        """Set up MV-Adapter attention processors and load weights."""
        from ..mvadapter.attention import set_mv_adapter_attn_processors
        
        # Determine base image size from pipeline config
        try:
            sample_size = pipeline.unet.config.sample_size
            base_img_size = sample_size  # 64 for SD, 96 for SDXL at 768px
        except:
            base_img_size = 96  # Default for SDXL
        
        print(f"[MV-Adapter] Setting up adapter with {num_views} views, base size {base_img_size}")
        
        # Set attention processors
        set_mv_adapter_attn_processors(
            pipeline.unet,
            num_views=num_views,
            base_img_size=base_img_size,
        )
        
        # Load adapter weights
        adapter_state_dict = self._load_adapter_weights(adapter_path)
        
        if adapter_state_dict:
            # Load weights into UNet
            missing, unexpected = pipeline.unet.load_state_dict(
                adapter_state_dict,
                strict=False,
            )
            
            if missing:
                print(f"[MV-Adapter] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[MV-Adapter] Unexpected keys: {len(unexpected)}")
            
            print(f"[MV-Adapter] Adapter weights loaded from {adapter_path}")
        
        # Store config for sampler
        pipeline._mvadapter_config = {
            "num_views": num_views,
            "adapter_mode": adapter_mode,
            "base_img_size": base_img_size,
        }
        
        return (pipeline,)
    
    def _load_adapter_weights(self, adapter_path: str) -> Dict[str, "torch.Tensor"]:
        """Load adapter weights from file or HuggingFace."""
        from safetensors.torch import load_file
        _ensure_imports()
        
        models_dir = get_mvadapter_models_dir()
        
        # Check if it's a local file
        local_path = os.path.join(models_dir, adapter_path)
        if os.path.exists(local_path):
            print(f"[MV-Adapter] Loading from local: {local_path}")
            return load_file(local_path)
        
        if os.path.exists(adapter_path):
            print(f"[MV-Adapter] Loading from path: {adapter_path}")
            return load_file(adapter_path)
        
        # Try HuggingFace
        if "/" in adapter_path:
            try:
                from huggingface_hub import hf_hub_download
                
                parts = adapter_path.split("/")
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    filename = "/".join(parts[2:])
                    
                    print(f"[MV-Adapter] Downloading from HuggingFace: {repo_id}/{filename}")
                    
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=get_mvadapter_models_dir(),
                    )
                    
                    from safetensors.torch import load_file
                    return load_file(downloaded_path)
            except Exception as e:
                print(f"[MV-Adapter] Error downloading from HuggingFace: {e}")
        
        print(f"[MV-Adapter] Warning: Could not load adapter from {adapter_path}")
        return {}


class MVAdapterLoRALoader:
    """
    Load LoRA weights into the MV-Adapter pipeline.
    
    Allows using custom styles and fine-tuned models with MV-Adapter.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pipeline": ("MVADAPTER_PIPELINE",),
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
            },
        }
    
    RETURN_TYPES = ("MVADAPTER_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_lora"
    CATEGORY = "MV-Adapter"
    
    def load_lora(
        self,
        pipeline,
        lora_path: str,
        lora_scale: float,
    ):
        """Load LoRA weights into the pipeline."""
        _ensure_imports()
        
        if not lora_path or not lora_path.strip():
            return (pipeline,)
        
        try:
            # Check if file exists in ComfyUI loras folder
            loras_dir = folder_paths.get_folder_paths("loras")[0]
            local_lora_path = os.path.join(loras_dir, lora_path)
            
            if os.path.exists(local_lora_path):
                lora_path = local_lora_path
            
            print(f"[MV-Adapter] Loading LoRA from {lora_path}")
            
            pipeline.load_lora_weights(lora_path)
            pipeline.fuse_lora(lora_scale=lora_scale)
            
            print(f"[MV-Adapter] LoRA loaded with scale {lora_scale}")
            
        except Exception as e:
            print(f"[MV-Adapter] Error loading LoRA: {e}")
        
        return (pipeline,)
