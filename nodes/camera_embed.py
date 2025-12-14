"""
Camera embedding node for MV-Adapter.

Generates Plücker coordinate embeddings for multi-view generation.
"""

import torch
from typing import Dict, Any, Tuple, Optional

from .pipeline_loader import get_torch_device


class MVAdapterCameraEmbed:
    """
    Generate camera embeddings for multi-view generation.
    
    Creates Plücker coordinate embeddings that encode camera positions
    for each view in the multi-view generation.
    """
    
    def __init__(self):
        self.device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "num_views": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 12,
                    "step": 1,
                }),
                "width": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
                "elevation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -90.0,
                    "max": 90.0,
                    "step": 5.0,
                }),
                "distance": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "azimuth_angles": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "0, 45, 90, 180, 270, 315",
                }),
            },
        }
    
    RETURN_TYPES = ("CAMERA_EMBED",)
    RETURN_NAMES = ("camera_embed",)
    FUNCTION = "generate_embeddings"
    CATEGORY = "MV-Adapter"
    
    def generate_embeddings(
        self,
        num_views: int,
        width: int,
        height: int,
        elevation_deg: float,
        distance: float,
        azimuth_angles: str = "",
    ) -> Tuple[Dict[str, Any]]:
        """Generate camera embeddings."""
        from ..utils.camera_utils import generate_camera_embeddings
        
        # Parse azimuth angles if provided
        azimuth_deg = None
        if azimuth_angles and azimuth_angles.strip():
            try:
                azimuth_deg = [float(a.strip()) for a in azimuth_angles.split(",")]
                if len(azimuth_deg) != num_views:
                    print(f"[MV-Adapter] Warning: {len(azimuth_deg)} angles provided but {num_views} views requested. Using default.")
                    azimuth_deg = None
            except ValueError:
                print("[MV-Adapter] Warning: Could not parse azimuth angles. Using default.")
                azimuth_deg = None
        
        # Generate camera embeddings
        plucker_embeds = generate_camera_embeddings(
            num_views=num_views,
            height=height,
            width=width,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            distance=distance,
        )
        
        print(f"[MV-Adapter] Generated camera embeddings: {plucker_embeds.shape}")
        
        # Package as dict for passing between nodes
        camera_embed = {
            "embeddings": plucker_embeds,
            "num_views": num_views,
            "width": width,
            "height": height,
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg or [i * (360.0 / num_views) for i in range(num_views)],
        }
        
        return (camera_embed,)


class MVAdapterViewSelector:
    """
    Select which views to generate.
    
    Provides a convenient way to configure standard view arrangements.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": ([
                    "6-view (standard)",
                    "4-view (cardinal)",
                    "8-view (full)",
                    "custom",
                ], {
                    "default": "6-view (standard)",
                }),
            },
            "optional": {
                "front": ("BOOLEAN", {"default": True}),
                "front_right": ("BOOLEAN", {"default": True}),
                "right": ("BOOLEAN", {"default": True}),
                "back": ("BOOLEAN", {"default": True}),
                "left": ("BOOLEAN", {"default": True}),
                "front_left": ("BOOLEAN", {"default": True}),
                "back_right": ("BOOLEAN", {"default": False}),
                "back_left": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("azimuth_angles", "num_views")
    FUNCTION = "select_views"
    CATEGORY = "MV-Adapter"
    
    # View name to azimuth angle mapping
    VIEW_ANGLES = {
        "front": 0,
        "front_right": 45,
        "right": 90,
        "back_right": 135,
        "back": 180,
        "back_left": 225,
        "left": 270,
        "front_left": 315,
    }
    
    def select_views(
        self,
        preset: str,
        front: bool = True,
        front_right: bool = True,
        right: bool = True,
        back: bool = True,
        left: bool = True,
        front_left: bool = True,
        back_right: bool = False,
        back_left: bool = False,
    ) -> Tuple[str, int]:
        """Select views and return azimuth angles."""
        
        if preset == "6-view (standard)":
            angles = [0, 45, 90, 180, 270, 315]
        elif preset == "4-view (cardinal)":
            angles = [0, 90, 180, 270]
        elif preset == "8-view (full)":
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            # Custom selection
            angles = []
            view_states = {
                "front": front,
                "front_right": front_right,
                "right": right,
                "back_right": back_right,
                "back": back,
                "back_left": back_left,
                "left": left,
                "front_left": front_left,
            }
            
            for view_name, enabled in view_states.items():
                if enabled:
                    angles.append(self.VIEW_ANGLES[view_name])
            
            # Sort by angle
            angles.sort()
        
        angles_str = ", ".join(str(a) for a in angles)
        num_views = len(angles)
        
        print(f"[MV-Adapter] Selected {num_views} views: {angles_str}")
        
        return (angles_str, num_views)
