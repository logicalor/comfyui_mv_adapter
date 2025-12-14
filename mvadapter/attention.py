"""
MV-Adapter attention processors for multi-view generation.

Based on: https://github.com/huanngzh/MV-Adapter
"""

import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class DecoupledMVRowSelfAttnProcessor2_0:
    """
    Attention processor for decoupled multi-view row self-attention.
    
    This processor implements the key innovation of MV-Adapter:
    - Performs self-attention within rows (same position across views)
    - Maintains multi-view consistency without cross-view attention overhead
    """
    
    def __init__(
        self,
        num_views: int = 6,
        base_img_size: int = 64,
    ):
        self.num_views = num_views
        self.base_img_size = base_img_size
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Multi-view row attention (only for self-attention)
        if not is_cross_attention:
            # Reshape for row attention across views
            # [B*num_views, heads, H*W, dim] -> [B, num_views, heads, H, W, dim]
            h = w = int((sequence_length) ** 0.5)
            
            if h * w == sequence_length:  # Only apply if square
                query = rearrange(
                    query, "(b v) h (ph pw) d -> b v h ph pw d",
                    v=self.num_views, ph=h, pw=w
                )
                key = rearrange(
                    key, "(b v) h (ph pw) d -> b v h ph pw d",
                    v=self.num_views, ph=h, pw=w
                )
                value = rearrange(
                    value, "(b v) h (ph pw) d -> b v h ph pw d",
                    v=self.num_views, ph=h, pw=w
                )
                
                # Row attention: attend across views at same row position
                query = rearrange(query, "b v h ph pw d -> (b ph) h (v pw) d")
                key = rearrange(key, "b v h ph pw d -> (b ph) h (v pw) d")
                value = rearrange(value, "b v h ph pw d -> (b ph) h (v pw) d")
                
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                
                # Reshape back
                hidden_states = rearrange(
                    hidden_states, "(b ph) h (v pw) d -> (b v) h (ph pw) d",
                    v=self.num_views, ph=h, pw=w
                )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def set_mv_adapter_attn_processors(
    unet,
    num_views: int = 6,
    base_img_size: int = 64,
):
    """
    Set MV-Adapter attention processors on a UNet model.
    
    Args:
        unet: The UNet model to modify
        num_views: Number of views to generate
        base_img_size: Base image size for attention (64 for 512px, 96 for 768px)
    """
    attn_procs = {}
    
    for name in unet.attn_processors.keys():
        if "attn1" in name:  # Self-attention layers only
            attn_procs[name] = DecoupledMVRowSelfAttnProcessor2_0(
                num_views=num_views,
                base_img_size=base_img_size,
            )
        else:
            # Keep original processor for cross-attention
            attn_procs[name] = unet.attn_processors[name]
    
    unet.set_attn_processor(attn_procs)
