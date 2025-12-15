"""
ComfyUI MV-Adapter Nodes

Multi-view image generation using MV-Adapter for ComfyUI.
Based on: https://github.com/huanngzh/MV-Adapter
"""

import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import (
        MVAdapterPipelineLoader,
        MVAdapterSchedulerConfig,
        MVAdapterModelSetup,
        MVAdapterLoRALoader,
        MVAdapterCameraEmbed,
        MVAdapterViewSelector,
        MVAdapterI2MVSampler,
        MVAdapterT2MVSampler,
        MVAdapterBackgroundRemoval,
        MVAdapterReferencePreprocess,
        MVAdapterImagePreprocess,
        MVAdapterImageGrid,
        MVAdapterSplitViews,
        MVAdapterClearVRAM,
        MVAdapterVAEDecode,
    )

    # Node class mappings for ComfyUI
    NODE_CLASS_MAPPINGS = {
        "MVAdapterPipelineLoader": MVAdapterPipelineLoader,
        "MVAdapterSchedulerConfig": MVAdapterSchedulerConfig,
        "MVAdapterModelSetup": MVAdapterModelSetup,
        "MVAdapterLoRALoader": MVAdapterLoRALoader,
        "MVAdapterCameraEmbed": MVAdapterCameraEmbed,
        "MVAdapterViewSelector": MVAdapterViewSelector,
        "MVAdapterI2MVSampler": MVAdapterI2MVSampler,
        "MVAdapterT2MVSampler": MVAdapterT2MVSampler,
        "MVAdapterBackgroundRemoval": MVAdapterBackgroundRemoval,
        "MVAdapterReferencePreprocess": MVAdapterReferencePreprocess,
        "MVAdapterImagePreprocess": MVAdapterImagePreprocess,
        "MVAdapterImageGrid": MVAdapterImageGrid,
        "MVAdapterSplitViews": MVAdapterSplitViews,
        "MVAdapterClearVRAM": MVAdapterClearVRAM,
        "MVAdapterVAEDecode": MVAdapterVAEDecode,
    }

    # Display names for nodes in ComfyUI UI
    NODE_DISPLAY_NAME_MAPPINGS = {
        "MVAdapterPipelineLoader": "MV-Adapter Pipeline Loader",
        "MVAdapterSchedulerConfig": "MV-Adapter Scheduler Config",
        "MVAdapterModelSetup": "MV-Adapter Model Setup",
        "MVAdapterLoRALoader": "MV-Adapter LoRA Loader",
        "MVAdapterCameraEmbed": "MV-Adapter Camera Embed",
        "MVAdapterViewSelector": "MV-Adapter View Selector",
        "MVAdapterI2MVSampler": "MV-Adapter I2MV Sampler",
        "MVAdapterT2MVSampler": "MV-Adapter T2MV Sampler",
        "MVAdapterBackgroundRemoval": "MV-Adapter Background Removal",
        "MVAdapterReferencePreprocess": "MV-Adapter Reference Preprocess (Official)",
        "MVAdapterImagePreprocess": "MV-Adapter Image Preprocess",
        "MVAdapterImageGrid": "MV-Adapter Image Grid",
        "MVAdapterSplitViews": "MV-Adapter Split Views",
        "MVAdapterClearVRAM": "MV-Adapter Clear VRAM",
        "MVAdapterVAEDecode": "MV-Adapter VAE Decode",
    }

except Exception as e:
    print(f"[MV-Adapter] Failed to load nodes: {e}")
    traceback.print_exc()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
