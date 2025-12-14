"""
ComfyUI MV-Adapter Nodes

Multi-view image generation using MV-Adapter for ComfyUI.
Based on: https://github.com/huanngzh/MV-Adapter

Nodes:
- MVAdapterPipelineLoader: Load base diffusion model
- MVAdapterSchedulerConfig: Configure scheduler with SNR shifting
- MVAdapterModelSetup: Initialize adapter attention processors
- MVAdapterLoRALoader: Load LoRA weights
- MVAdapterCameraEmbed: Generate camera embeddings
- MVAdapterViewSelector: Select which views to generate
- MVAdapterI2MVSampler: Image-to-Multiview generation
- MVAdapterT2MVSampler: Text-to-Multiview generation
- MVAdapterBackgroundRemoval: Remove background from images
- MVAdapterImagePreprocess: Preprocess images for generation
- MVAdapterImageGrid: Combine views into grid
- MVAdapterSplitViews: Split batch into individual views
"""

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
    MVAdapterImagePreprocess,
    MVAdapterImageGrid,
    MVAdapterSplitViews,
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
    "MVAdapterImagePreprocess": MVAdapterImagePreprocess,
    "MVAdapterImageGrid": MVAdapterImageGrid,
    "MVAdapterSplitViews": MVAdapterSplitViews,
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
    "MVAdapterImagePreprocess": "MV-Adapter Image Preprocess",
    "MVAdapterImageGrid": "MV-Adapter Image Grid",
    "MVAdapterSplitViews": "MV-Adapter Split Views",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
