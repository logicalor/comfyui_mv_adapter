"""
Scheduler utilities for MV-Adapter.

Implements ShiftSNR scheduler for improved multi-view generation quality.
Based on the official MV-Adapter implementation:
https://github.com/huanngzh/MV-Adapter/blob/main/mvadapter/schedulers/scheduling_shift_snr.py
"""

import torch
import numpy as np
from typing import Any, Optional


def compute_snr(timesteps: torch.Tensor, noise_scheduler) -> torch.Tensor:
    """
    Compute SNR (Signal-to-Noise Ratio) for given timesteps.
    
    Based on Min-SNR-Diffusion-Training approach.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()

    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod

    # Compute SNR
    snr = (alpha / sigma) ** 2
    return snr


def SNR_to_betas(snr: torch.Tensor) -> torch.Tensor:
    """
    Convert SNR values back to betas for the scheduler.
    """
    # alpha_t^2 / (1 - alpha_t^2) = snr
    # alpha_t = sqrt(snr / (1 + snr))
    alpha_t = (snr / (1 + snr)) ** 0.5
    alphas_cumprod = alpha_t ** 2
    
    # alphas = alphas_cumprod[t] / alphas_cumprod[t-1]
    alphas = alphas_cumprod / torch.cat([torch.ones(1, device=snr.device), alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas


class ShiftSNRScheduler:
    """
    Factory class that creates schedulers with shifted SNR noise schedules.
    
    This matches the official MV-Adapter implementation which creates a new
    scheduler with modified betas rather than wrapping an existing one.
    """
    
    def __init__(
        self,
        noise_scheduler: Any,
        timesteps: torch.Tensor,
        shift_scale: float,
        scheduler_class: Any,
    ):
        self.noise_scheduler = noise_scheduler
        self.timesteps = timesteps
        self.shift_scale = shift_scale
        self.scheduler_class = scheduler_class
    
    def _get_shift_scheduler(self):
        """
        Create scheduler with shifted betas (simple scaling).
        """
        snr = compute_snr(self.timesteps, self.noise_scheduler)
        shifted_betas = SNR_to_betas(snr / self.shift_scale)
        
        return self.scheduler_class.from_config(
            self.noise_scheduler.config, 
            trained_betas=shifted_betas.numpy()
        )
    
    def _get_interpolated_shift_scheduler(self):
        """
        Create scheduler with interpolated shifted betas.
        
        Interpolates between original and shifted SNR in log space for
        smoother transition across timesteps.
        """
        snr = compute_snr(self.timesteps, self.noise_scheduler)
        shifted_snr = snr / self.shift_scale
        
        # Weight increases linearly from 0 to 1 across timesteps
        weighting = self.timesteps.float() / (self.noise_scheduler.config.num_train_timesteps - 1)
        
        # Interpolate in log space
        interpolated_snr = torch.exp(
            torch.log(snr) * (1 - weighting) + torch.log(shifted_snr) * weighting
        )
        
        shifted_betas = SNR_to_betas(interpolated_snr)
        
        return self.scheduler_class.from_config(
            self.noise_scheduler.config,
            trained_betas=shifted_betas.numpy()
        )
    
    @classmethod
    def from_scheduler(
        cls,
        noise_scheduler: Any,
        shift_mode: str = "interpolated",
        timesteps: Optional[torch.Tensor] = None,
        shift_scale: float = 8.0,
        scheduler_class: Optional[Any] = None,
    ):
        """
        Create a new scheduler with SNR shifting applied.
        
        Args:
            noise_scheduler: Base scheduler to derive config from
            shift_mode: "default" for simple shift, "interpolated" for smooth transition
            timesteps: Timesteps to use (defaults to full range)
            shift_scale: Scale factor for SNR shifting (default 8.0)
            scheduler_class: Class to use for new scheduler (defaults to same as input)
            
        Returns:
            New scheduler instance with shifted noise schedule
        """
        if timesteps is None:
            timesteps = torch.arange(0, noise_scheduler.config.num_train_timesteps)
        if scheduler_class is None:
            scheduler_class = noise_scheduler.__class__
        
        shift_scheduler = cls(
            noise_scheduler=noise_scheduler,
            timesteps=timesteps,
            shift_scale=shift_scale,
            scheduler_class=scheduler_class,
        )
        
        if shift_mode == "default":
            return shift_scheduler._get_shift_scheduler()
        elif shift_mode == "interpolated":
            return shift_scheduler._get_interpolated_shift_scheduler()
        else:
            raise ValueError(f"Unknown shift_mode: {shift_mode}")


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
        shift_mode: Shift mode ("default" or "interpolated")
        shift_scale: Scale factor for shifting (default 8.0)
        
    Returns:
        Scheduler with SNR shift applied (if enabled)
    """
    if shift_snr:
        return ShiftSNRScheduler.from_scheduler(
            noise_scheduler=scheduler,
            shift_mode=shift_mode,
            shift_scale=shift_scale,
        )
    return scheduler
