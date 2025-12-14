"""
Scheduler utilities for MV-Adapter.

Implements ShiftSNR scheduler wrapper for improved multi-view generation quality.
"""

import torch
import numpy as np
from typing import Optional


class ShiftSNRScheduler:
    """
    Wrapper that applies SNR shifting to any scheduler.
    
    SNR (Signal-to-Noise Ratio) shifting improves the quality of multi-view
    generation by adjusting the noise schedule.
    """
    
    def __init__(
        self,
        scheduler,
        shift_mode: str = "interpolated",
        shift_scale: float = 8.0,
        scheduler_class: Optional[type] = None,
    ):
        """
        Args:
            scheduler: The base scheduler to wrap
            shift_mode: How to shift SNR ("interpolated" or "scaled")
            shift_scale: Scale factor for shifting
            scheduler_class: Original scheduler class (for compatibility)
        """
        self.scheduler = scheduler
        self.shift_mode = shift_mode
        self.shift_scale = shift_scale
        self.scheduler_class = scheduler_class or type(scheduler)
        
        # Copy attributes from base scheduler
        self._copy_scheduler_attributes()
        
        # Apply SNR shift
        self._apply_snr_shift()
    
    def _copy_scheduler_attributes(self):
        """Copy important attributes from base scheduler."""
        attrs_to_copy = [
            'num_train_timesteps', 'timesteps', 'alphas_cumprod',
            'final_alpha_cumprod', 'init_noise_sigma', 'config'
        ]
        for attr in attrs_to_copy:
            if hasattr(self.scheduler, attr):
                setattr(self, attr, getattr(self.scheduler, attr))
    
    def _apply_snr_shift(self):
        """Apply SNR shifting to the noise schedule."""
        if not hasattr(self.scheduler, 'alphas_cumprod'):
            return
        
        alphas_cumprod = self.scheduler.alphas_cumprod.clone()
        
        if self.shift_mode == "interpolated":
            # Interpolated shift: smoother transition
            snr = alphas_cumprod / (1 - alphas_cumprod)
            shifted_snr = snr / self.shift_scale
            shifted_alphas_cumprod = shifted_snr / (1 + shifted_snr)
            
            # Interpolate between original and shifted
            t = torch.linspace(0, 1, len(alphas_cumprod))
            self.scheduler.alphas_cumprod = (
                (1 - t) * alphas_cumprod + t * shifted_alphas_cumprod
            )
        
        elif self.shift_mode == "scaled":
            # Simple scaling
            snr = alphas_cumprod / (1 - alphas_cumprod)
            shifted_snr = snr / self.shift_scale
            self.scheduler.alphas_cumprod = shifted_snr / (1 + shifted_snr)
    
    def set_timesteps(self, num_inference_steps: int, device=None):
        """Set timesteps for inference."""
        return self.scheduler.set_timesteps(num_inference_steps, device=device)
    
    def step(self, *args, **kwargs):
        """Perform one scheduler step."""
        return self.scheduler.step(*args, **kwargs)
    
    def add_noise(self, *args, **kwargs):
        """Add noise to samples."""
        return self.scheduler.add_noise(*args, **kwargs)
    
    def scale_model_input(self, *args, **kwargs):
        """Scale model input."""
        return self.scheduler.scale_model_input(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to base scheduler."""
        return getattr(self.scheduler, name)


def create_scheduler_with_shift(
    scheduler,
    shift_snr: bool = True,
    shift_mode: str = "interpolated",
    shift_scale: float = 8.0,
):
    """
    Create a scheduler with optional SNR shifting.
    
    Args:
        scheduler: Base scheduler instance
        shift_snr: Whether to apply SNR shifting
        shift_mode: Shift mode ("interpolated" or "scaled")
        shift_scale: Scale factor for shifting
        
    Returns:
        Scheduler (wrapped with SNR shift if enabled)
    """
    if shift_snr:
        return ShiftSNRScheduler(
            scheduler=scheduler,
            shift_mode=shift_mode,
            shift_scale=shift_scale,
        )
    return scheduler
