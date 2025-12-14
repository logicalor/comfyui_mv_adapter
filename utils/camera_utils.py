"""
Camera utilities for MV-Adapter.
Generates Plücker coordinate embeddings for multi-view generation.

Based on: https://github.com/huanngzh/MV-Adapter
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Optional


def get_orthogonal_camera(
    elevation_deg: float,
    distance: float,
    left: float = -1.0,
    right: float = 1.0,
    bottom: float = -1.0,
    top: float = 1.0,
    azimuth_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate orthogonal camera extrinsic and intrinsic matrices.
    
    Args:
        elevation_deg: Camera elevation angle in degrees
        distance: Distance from camera to origin
        left, right, bottom, top: Orthographic projection bounds
        azimuth_deg: Camera azimuth angle in degrees
        
    Returns:
        Tuple of (extrinsic matrix 4x4, intrinsic matrix 4x4)
    """
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    
    # Camera position in spherical coordinates
    x = distance * math.cos(elevation) * math.sin(azimuth)
    y = distance * math.sin(elevation)
    z = distance * math.cos(elevation) * math.cos(azimuth)
    
    camera_pos = np.array([x, y, z])
    
    # Look at origin
    forward = -camera_pos / np.linalg.norm(camera_pos)
    
    # Up vector (world Y)
    world_up = np.array([0.0, 1.0, 0.0])
    
    # Handle edge case when looking straight up/down
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    
    right_vec = np.cross(forward, world_up)
    right_vec = right_vec / np.linalg.norm(right_vec)
    
    up_vec = np.cross(right_vec, forward)
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    # Extrinsic matrix (camera to world)
    extrinsic = np.eye(4)
    extrinsic[:3, 0] = right_vec
    extrinsic[:3, 1] = up_vec
    extrinsic[:3, 2] = -forward
    extrinsic[:3, 3] = camera_pos
    
    # Orthographic intrinsic matrix
    intrinsic = np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    return extrinsic, intrinsic


def get_plucker_embeds_from_cameras_ortho(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Generate Plücker coordinate embeddings from camera matrices.
    
    Plücker coordinates represent rays as 6D vectors (direction, moment).
    
    Args:
        extrinsics: Camera extrinsic matrices [N, 4, 4]
        intrinsics: Camera intrinsic matrices [N, 4, 4]
        width: Image width
        height: Image height
        
    Returns:
        Plücker embeddings tensor [N, H, W, 6]
    """
    num_views = extrinsics.shape[0]
    
    # Create pixel grid
    u = np.linspace(0.5, width - 0.5, width)
    v = np.linspace(0.5, height - 0.5, height)
    u, v = np.meshgrid(u, v)
    
    # Normalize to [-1, 1]
    u_norm = (u / width) * 2 - 1
    v_norm = (v / height) * 2 - 1
    
    plucker_embeds = []
    
    for i in range(num_views):
        ext = extrinsics[i]
        
        # Camera position (origin of rays for orthographic)
        cam_pos = ext[:3, 3]
        
        # Ray direction (same for all pixels in orthographic)
        ray_dir = -ext[:3, 2]  # Forward direction
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # For orthographic camera, ray origins are offset by pixel position
        right_vec = ext[:3, 0]
        up_vec = ext[:3, 1]
        
        # Calculate ray origins for each pixel
        ray_origins = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                offset = u_norm[y, x] * right_vec + v_norm[y, x] * up_vec
                ray_origins[y, x] = cam_pos + offset
        
        # Plücker coordinates: (direction, moment)
        # moment = origin x direction
        ray_dirs = np.broadcast_to(ray_dir, (height, width, 3))
        moments = np.cross(ray_origins, ray_dirs)
        
        # Concatenate direction and moment
        plucker = np.concatenate([ray_dirs, moments], axis=-1)
        plucker_embeds.append(plucker)
    
    plucker_embeds = np.stack(plucker_embeds, axis=0)
    return torch.from_numpy(plucker_embeds).float()


def generate_camera_embeddings(
    num_views: int,
    height: int,
    width: int,
    azimuth_deg: Optional[List[float]] = None,
    elevation_deg: float = 0.0,
    distance: float = 1.5,
) -> torch.Tensor:
    """
    Generate camera embeddings for multi-view generation.
    
    Args:
        num_views: Number of views to generate
        height: Image height
        width: Image width
        azimuth_deg: List of azimuth angles in degrees. 
                     If None, uses evenly spaced angles.
        elevation_deg: Elevation angle in degrees (same for all views)
        distance: Camera distance from origin
        
    Returns:
        Camera embeddings tensor [N, H, W, 6]
    """
    if azimuth_deg is None:
        # Default: evenly spaced views starting from front
        azimuth_deg = [i * (360.0 / num_views) for i in range(num_views)]
    
    assert len(azimuth_deg) == num_views, \
        f"Number of azimuth angles ({len(azimuth_deg)}) must match num_views ({num_views})"
    
    extrinsics = []
    intrinsics = []
    
    for azimuth in azimuth_deg:
        ext, intr = get_orthogonal_camera(
            elevation_deg=elevation_deg,
            distance=distance,
            azimuth_deg=azimuth,
        )
        extrinsics.append(ext)
        intrinsics.append(intr)
    
    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        extrinsics, intrinsics, width, height
    )
    
    return plucker_embeds


def prepare_camera_embed_for_pipeline(
    plucker_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepare camera embeddings for the diffusion pipeline.
    
    Converts from [N, H, W, 6] to [N, 6, H, W] format expected by the model.
    
    Args:
        plucker_embeds: Camera embeddings [N, H, W, 6]
        device: Target device
        dtype: Target dtype
        
    Returns:
        Camera embeddings tensor [N, 6, H, W]
    """
    # Rearrange from NHWC to NCHW
    embeds = plucker_embeds.permute(0, 3, 1, 2)
    return embeds.to(device=device, dtype=dtype)


# Default view configurations
DEFAULT_VIEWS_6 = {
    "azimuths": [0, 45, 90, 180, 270, 315],
    "names": ["front", "front_right", "right", "back", "left", "front_left"]
}

DEFAULT_VIEWS_4 = {
    "azimuths": [0, 90, 180, 270],
    "names": ["front", "right", "back", "left"]
}

DEFAULT_VIEWS_8 = {
    "azimuths": [0, 45, 90, 135, 180, 225, 270, 315],
    "names": ["front", "front_right", "right", "back_right", 
              "back", "back_left", "left", "front_left"]
}
