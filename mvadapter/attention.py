"""
MV-Adapter attention processors for multi-view generation.

Adapted from: https://github.com/huanngzh/MV-Adapter
Licensed under Apache 2.0
"""

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.unets import UNet2DConditionModel
from diffusers.utils import deprecate, logging
from einops import rearrange

logger = logging.get_logger(__name__)


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_unet_2d_condition_attn_processor(
    unet: UNet2DConditionModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_proc_func: Callable = default_set_attn_proc_func,
    set_custom_attn_proc_func: Callable = default_set_attn_proc_func,
    set_self_attn_module_names: Optional[List[str]] = None,
    set_cross_attn_module_names: Optional[List[str]] = None,
    set_custom_attn_module_names: Optional[List[str]] = None,
) -> None:
    """Set attention processors on a UNet model."""
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )

    attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        set_self_attn_processor = do_set_processor(name, set_self_attn_module_names)
        set_cross_attn_processor = do_set_processor(name, set_cross_attn_module_names)
        set_custom_attn_processor = do_set_processor(name, set_custom_attn_module_names)

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        is_custom = "attn_mid_blocks" in name or "attn_post_blocks" in name
        if is_custom:
            attn_procs[name] = (
                set_custom_attn_proc_func(name, hidden_size, None, attn_processor)
                if set_custom_attn_processor
                else attn_processor
            )
        else:
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if cross_attention_dim is None or "motion_modules" in name:
                # self attention
                attn_procs[name] = (
                    set_self_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_self_attn_processor
                    else attn_processor
                )
            else:
                # cross attention
                attn_procs[name] = (
                    set_cross_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_cross_attn_processor
                    else attn_processor
                )

    unet.set_attn_processor(attn_procs)


class DecoupledMVRowSelfAttnProcessor2_0(nn.Module):
    """
    Attention processor for Decoupled Row-wise Self-Attention and Image Cross-Attention.
    
    This is the core attention mechanism for MV-Adapter that enables:
    - Multi-view consistent attention across views (row-wise)
    - Reference image cross-attention for I2MV generation
    """

    def __init__(
        self,
        query_dim: int,
        inner_dim: int,
        num_views: int = 1,
        name: Optional[str] = None,
        use_mv: bool = True,
        use_ref: bool = False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "DecoupledMVRowSelfAttnProcessor2_0 requires PyTorch 2.0"
            )

        super().__init__()

        self.num_views = num_views
        self.name = name
        self.use_mv = use_mv
        self.use_ref = use_ref

        if self.use_mv:
            self.to_q_mv = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k_mv = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_v_mv = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_out_mv = nn.ModuleList([
                nn.Linear(inner_dim, query_dim, bias=True),
                nn.Dropout(0.0),
            ])

        if self.use_ref:
            self.to_q_ref = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k_ref = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_v_ref = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_out_ref = nn.ModuleList([
                nn.Linear(inner_dim, query_dim, bias=True),
                nn.Dropout(0.0),
            ])

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        mv_scale: float = 1.0,
        ref_hidden_states: Optional[dict] = None,
        ref_scale: float = 1.0,
        cache_hidden_states: Optional[dict] = None,
        use_mv: bool = True,
        use_ref: bool = True,
        num_views: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated."
            deprecate("scale", "1.0.0", deprecation_message)

        if num_views is not None:
            self.num_views = num_views

        # Cache hidden states for reference unet
        if cache_hidden_states is not None:
            cache_hidden_states[self.name] = hidden_states.clone()

        # Whether to use multi-view attention and image cross-attention
        use_mv = self.use_mv and use_mv
        use_ref = self.use_ref and use_ref

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # For decoupled multi-view attention
        if use_mv:
            query_mv = self.to_q_mv(hidden_states)

        # For decoupled reference cross attention
        if use_ref:
            query_ref = self.to_q_ref(hidden_states)

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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Decoupled multi-view self-attention
        if use_mv:
            key_mv = self.to_k_mv(encoder_hidden_states)
            value_mv = self.to_v_mv(encoder_hidden_states)

            query_mv = query_mv.view(batch_size, -1, attn.heads, head_dim)
            key_mv = key_mv.view(batch_size, -1, attn.heads, head_dim)
            value_mv = value_mv.view(batch_size, -1, attn.heads, head_dim)

            height = width = math.isqrt(sequence_length)

            # Row self-attention
            query_mv = rearrange(
                query_mv,
                "(b nv) (ih iw) h c -> (b nv ih) iw h c",
                nv=self.num_views, ih=height, iw=width,
            ).transpose(1, 2)
            
            key_mv = rearrange(
                key_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views, ih=height, iw=width,
            )
            key_mv = (
                key_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            
            value_mv = rearrange(
                value_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views, ih=height, iw=width,
            )
            value_mv = (
                value_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )

            hidden_states_mv = F.scaled_dot_product_attention(
                query_mv, key_mv, value_mv, dropout_p=0.0, is_causal=False,
            )
            hidden_states_mv = rearrange(
                hidden_states_mv,
                "(b nv ih) h iw c -> (b nv) (ih iw) (h c)",
                nv=self.num_views, ih=height,
            )
            hidden_states_mv = hidden_states_mv.to(query.dtype)

            # Linear proj
            hidden_states_mv = self.to_out_mv[0](hidden_states_mv)
            # Dropout
            hidden_states_mv = self.to_out_mv[1](hidden_states_mv)

        if use_ref and ref_hidden_states is not None:
            reference_hidden_states = ref_hidden_states[self.name]
            
            # Ensure reference hidden states are on the same device as query
            reference_hidden_states = reference_hidden_states.to(device=query.device, dtype=query.dtype)

            key_ref = self.to_k_ref(reference_hidden_states)
            value_ref = self.to_v_ref(reference_hidden_states)

            query_ref = query_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_ref = key_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ref = value_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_ref = F.scaled_dot_product_attention(
                query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False
            )

            hidden_states_ref = hidden_states_ref.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states_ref = hidden_states_ref.to(query.dtype)

            # Linear proj
            hidden_states_ref = self.to_out_ref[0](hidden_states_ref)
            # Dropout
            hidden_states_ref = self.to_out_ref[1](hidden_states_ref)

        # Linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if use_mv:
            hidden_states = hidden_states + hidden_states_mv * mv_scale

        if use_ref and ref_hidden_states is not None:
            hidden_states = hidden_states + hidden_states_ref * ref_scale

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def set_num_views(self, num_views: int) -> None:
        self.num_views = num_views


# Simplified wrapper for backwards compatibility
def set_mv_adapter_attn_processors(
    unet,
    num_views: int = 6,
    base_img_size: int = 64,
):
    """
    Set MV-Adapter attention processors on a UNet model.
    Simplified version - use set_unet_2d_condition_attn_processor for full control.
    """
    set_unet_2d_condition_attn_processor(
        unet,
        set_self_attn_proc_func=lambda name, hs, cad, ap: DecoupledMVRowSelfAttnProcessor2_0(
            query_dim=hs,
            inner_dim=hs,
            num_views=num_views,
            name=name,
            use_mv=True,
            use_ref=True,
        ),
        set_cross_attn_proc_func=lambda name, hs, cad, ap: DecoupledMVRowSelfAttnProcessor2_0(
            query_dim=hs,
            inner_dim=hs,
            num_views=num_views,
            name=name,
            use_mv=False,
            use_ref=False,
        ),
    )
