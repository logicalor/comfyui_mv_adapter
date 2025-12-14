"""
MV-Adapter core components.

Bundled from https://github.com/huanngzh/MV-Adapter to avoid torch reinstall.
"""

from .attention import (
    DecoupledMVRowSelfAttnProcessor2_0,
    set_mv_adapter_attn_processors,
    set_unet_2d_condition_attn_processor,
)

from .loaders import CustomAdapterMixin

from .pipelines import MVAdapterI2MVSDXLPipeline

from .scheduler import (
    ShiftSNRScheduler,
    create_scheduler_with_shift,
)

__all__ = [
    "DecoupledMVRowSelfAttnProcessor2_0",
    "set_mv_adapter_attn_processors",
    "set_unet_2d_condition_attn_processor",
    "CustomAdapterMixin",
    "MVAdapterI2MVSDXLPipeline",
    "ShiftSNRScheduler",
    "create_scheduler_with_shift",
]
