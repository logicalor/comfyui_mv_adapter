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
    
    # Available adapters from HuggingFace
    AVAILABLE_ADAPTERS = {
        "I2MV SDXL": "huanngzh/mv-adapter/mvadapter_i2mv_sdxl.safetensors",
        "T2MV SDXL": "huanngzh/mv-adapter/mvadapter_t2mv_sdxl.safetensors",
        "I2MV SD2.1": "huanngzh/mv-adapter/mvadapter_i2mv_sd21.safetensors",
        "T2MV SD2.1": "huanngzh/mv-adapter/mvadapter_t2mv_sd21.safetensors",
    }
    
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
                "auto_download": ("BOOLEAN", {
                    "default": True,
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
        auto_download: bool = True,
    ) -> Tuple[Any]:
        """Set up MV-Adapter using the init_custom_adapter and load_custom_adapter methods."""
        _ensure_imports()
        
        # Import the attention processor from our bundled code
        from ..mvadapter.attention import DecoupledMVRowSelfAttnProcessor2_0
        
        print(f"[MV-Adapter] Initializing adapter with {num_views} views")
        
        # Initialize the custom adapter (sets up cond_encoder and attention processors)
        pipeline.init_custom_adapter(num_views=num_views)
        
        # Resolve and download adapter weights path
        adapter_file_path = self._resolve_adapter_path(adapter_path, auto_download)
        
        # Load adapter weights using official method
        if adapter_file_path:
            adapter_dir = os.path.dirname(adapter_file_path)
            adapter_filename = os.path.basename(adapter_file_path)
            
            print(f"[MV-Adapter] Loading adapter weights: {adapter_filename}")
            pipeline.load_custom_adapter(adapter_dir, weight_name=adapter_filename)
            print(f"[MV-Adapter] Adapter weights loaded successfully")
        
        # Move cond_encoder to device and dtype
        if hasattr(pipeline, 'cond_encoder'):
            pipeline.cond_encoder.to(device=self.device, dtype=pipeline.dtype)
            print(f"[MV-Adapter] cond_encoder moved to {self.device}")
        
        # Ensure all attention processors are on the correct device
        for name, processor in pipeline.unet.attn_processors.items():
            if hasattr(processor, 'to'):
                processor.to(device=self.device, dtype=pipeline.dtype)
        
        # Store config for sampler
        pipeline._mvadapter_config = {
            "num_views": num_views,
            "adapter_mode": adapter_mode,
        }
        
        return (pipeline,)
    
    def _resolve_adapter_path(self, adapter_path: str, auto_download: bool = True) -> str:
        """Resolve adapter path - download from HuggingFace if needed."""
        _ensure_imports()
        
        models_dir = get_mvadapter_models_dir()
        
        # Check if it's a local file in mvadapter models dir
        local_path = os.path.join(models_dir, adapter_path)
        if os.path.exists(local_path):
            print(f"[MV-Adapter] Using local adapter: {local_path}")
            return local_path
        
        # Check if it's an absolute/relative path that exists
        if os.path.exists(adapter_path):
            print(f"[MV-Adapter] Using adapter at: {adapter_path}")
            return adapter_path
        
        # Try HuggingFace download
        if "/" in adapter_path:
            if not auto_download:
                raise ValueError(
                    f"Adapter '{adapter_path}' not found locally and auto_download is disabled. "
                    f"Enable auto_download or download manually to: {models_dir}"
                )
            
            try:
                from huggingface_hub import hf_hub_download
                
                parts = adapter_path.split("/")
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    filename = "/".join(parts[2:])
                    
                    print(f"[MV-Adapter] Downloading from HuggingFace: {repo_id}/{filename}")
                    print(f"[MV-Adapter] This may take a while on first run...")
                    
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=models_dir,
                    )
                    
                    print(f"[MV-Adapter] Download complete: {downloaded_path}")
                    return downloaded_path
                else:
                    raise ValueError(
                        f"Invalid HuggingFace path format: '{adapter_path}'. "
                        f"Expected format: 'owner/repo/filename.safetensors'"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download adapter from HuggingFace: {e}\n"
                    f"You can manually download from https://huggingface.co/{adapter_path.rsplit('/', 1)[0]}"
                ) from e
        
        raise FileNotFoundError(
            f"Adapter not found: '{adapter_path}'\n"
            f"Place adapter files in: {models_dir}\n"
            f"Or use a HuggingFace path like: huanngzh/mv-adapter/mvadapter_i2mv_sdxl.safetensors"
        )


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
