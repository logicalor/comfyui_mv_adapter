"""
MV-Adapter Pipeline implementations for multi-view generation.

Bundled from: https://github.com/huanngzh/MV-Adapter
Licensed under Apache 2.0

This file contains the full pipeline implementation to avoid installing
the MV-Adapter package which would reinstall torch (breaking ROCm).
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, T2IAdapter, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from .loaders import CustomAdapterMixin
from .attention import DecoupledMVRowSelfAttnProcessor2_0, set_unet_2d_condition_attn_processor

logger = logging.get_logger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class MVAdapterI2MVSDXLPipeline(StableDiffusionXLPipeline, CustomAdapterMixin):
    """
    MV-Adapter pipeline for Image-to-Multi-View generation using SDXL.
    
    This pipeline extends StableDiffusionXLPipeline with multi-view attention
    and reference image conditioning capabilities.
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def prepare_image_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        add_noise=True,
    ):
        """Prepare latents from an image."""
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            if latents_mean is not None and latents_std is not None:
                latents_mean = latents_mean.to(device=device, dtype=dtype)
                latents_std = latents_std.to(device=device, dtype=dtype)
                init_latents = (
                    (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
                )
            else:
                init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents
        return latents

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        num_empty_images=0,
    ):
        """Prepare control image (camera embeddings) for conditioning."""
        assert hasattr(self, "control_image_processor"), "control_image_processor is not initialized"

        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        if num_empty_images > 0:
            image = torch.cat([image, torch.zeros_like(image[:num_empty_images])], dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # MV-Adapter specific parameters
        mv_scale: float = 1.0,
        control_image: Optional[PipelineImageInput] = None,
        control_conditioning_scale: Optional[float] = 1.0,
        control_conditioning_factor: float = 1.0,
        reference_image: Optional[PipelineImageInput] = None,
        reference_conditioning_scale: Optional[float] = 1.0,
        **kwargs,
    ):
        """Generate multi-view images from a reference image."""
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # Preprocess reference image
        print(f"[MV-Adapter Debug] reference_image type: {type(reference_image)}")
        if hasattr(reference_image, 'size'):
            print(f"[MV-Adapter Debug] reference_image size: {reference_image.size}")
        reference_image = self.image_processor.preprocess(reference_image)
        print(f"[MV-Adapter Debug] preprocessed reference_image shape: {reference_image.shape}")
        reference_latents = self.prepare_image_latents(
            reference_image,
            timesteps[:1].repeat(batch_size * num_images_per_prompt),
            batch_size,
            1,
            prompt_embeds.dtype,
            device,
            generator,
            add_noise=False,
        )
        print(f"[MV-Adapter Debug] reference_latents shape: {reference_latents.shape}")

        with torch.no_grad():
            ref_timesteps = torch.zeros_like(timesteps[0])
            ref_hidden_states = {}
            
            # Use the positive prompt embeddings (last one if CFG is used)
            ref_prompt_embeds = prompt_embeds[-1:] if self.do_classifier_free_guidance else prompt_embeds[:1]
            ref_text_embeds = add_text_embeds[-1:] if self.do_classifier_free_guidance else add_text_embeds[:1]
            ref_time_ids = add_time_ids[-1:] if self.do_classifier_free_guidance else add_time_ids[:1]
            
            print(f"[MV-Adapter Debug] Running reference UNet pass...")
            print(f"[MV-Adapter Debug] ref_prompt_embeds shape: {ref_prompt_embeds.shape}")

            self.unet(
                reference_latents,
                ref_timesteps,
                encoder_hidden_states=ref_prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": ref_text_embeds,
                    "time_ids": ref_time_ids,
                },
                cross_attention_kwargs={
                    "cache_hidden_states": ref_hidden_states,
                    "use_mv": False,
                    "use_ref": False,
                },
                return_dict=False,
            )
            print(f"[MV-Adapter Debug] ref_hidden_states populated with {len(ref_hidden_states)} keys")
            ref_hidden_states = {
                k: v.repeat_interleave(num_images_per_prompt, dim=0).to(device=device)
                for k, v in ref_hidden_states.items()
            }
        if self.do_classifier_free_guidance:
            ref_hidden_states = {
                k: torch.cat([torch.zeros_like(v), v], dim=0).to(device=device)
                for k, v in ref_hidden_states.items()
            }

        # Debug: Log reference hidden states info
        print(f"[MV-Adapter Debug] ref_hidden_states keys: {list(ref_hidden_states.keys())[:5]}... ({len(ref_hidden_states)} total)")
        if ref_hidden_states:
            first_key = list(ref_hidden_states.keys())[0]
            print(f"[MV-Adapter Debug] ref_hidden_states['{first_key}'] shape: {ref_hidden_states[first_key].shape}")
        else:
            print(f"[MV-Adapter Warning] ref_hidden_states is EMPTY! Reference conditioning will not work.")
        print(f"[MV-Adapter Debug] reference_conditioning_scale (ref_scale): {reference_conditioning_scale}")
        print(f"[MV-Adapter Debug] mv_scale: {mv_scale}")

        cross_attention_kwargs = {
            "mv_scale": mv_scale,
            "ref_hidden_states": {k: v.clone().to(device=device) for k, v in ref_hidden_states.items()},
            "ref_scale": reference_conditioning_scale,
            "num_views": num_images_per_prompt,
            "use_mv": True,
            "use_ref": True,
            **(self.cross_attention_kwargs or {}),
        }

        # Preprocess control image
        control_image_feature = self.prepare_control_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=1,
            device=device,
            dtype=latents.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        control_image_feature = control_image_feature.to(device=device, dtype=latents.dtype)

        # Ensure cond_encoder is on correct device before running
        self.cond_encoder.to(device=device, dtype=latents.dtype)
        
        adapter_state = self.cond_encoder(control_image_feature)
        for i, state in enumerate(adapter_state):
            adapter_state[i] = state.to(device=device) * control_conditioning_scale

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                if i < int(num_inference_steps * control_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(latents.dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            # Memory-efficient VAE decode: decode one image at a time to reduce peak memory
            # This is crucial for multi-view generation where we have many latents
            if latents.shape[0] > 1:
                images = []
                for i in range(latents.shape[0]):
                    single_latent = latents[i:i+1]
                    single_image = self.vae.decode(single_latent, return_dict=False)[0]
                    images.append(single_image)
                    # Clear cache between decodes
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                image = torch.cat(images, dim=0)
                del images
            else:
                image = self.vae.decode(latents, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        
        # Clean up memory
        del cross_attention_kwargs
        del adapter_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def _init_custom_adapter(
        self,
        num_views: int = 1,
        self_attn_processor: Any = DecoupledMVRowSelfAttnProcessor2_0,
        cond_in_channels: int = 6,
        copy_attn_weights: bool = True,
        zero_init_module_keys: List[str] = [],
    ):
        """Initialize the MV-Adapter with custom attention processors and condition encoder."""
        # Get device and dtype from unet
        device = self.unet.device
        dtype = self.unet.dtype
        
        # Condition encoder (T2I-Adapter)
        self.cond_encoder = T2IAdapter(
            in_channels=cond_in_channels,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        ).to(device=device, dtype=dtype)

        # Set custom attn processor for multi-view attention and image cross-attention
        self.unet: UNet2DConditionModel
        set_unet_2d_condition_attn_processor(
            self.unet,
            set_self_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=True,
                use_ref=True,
            ).to(device=device, dtype=dtype),
            set_cross_attn_proc_func=lambda name, hs, cad, ap: self_attn_processor(
                query_dim=hs,
                inner_dim=hs,
                num_views=num_views,
                name=name,
                use_mv=False,
                use_ref=False,
            ).to(device=device, dtype=dtype),
        )

        # Copy decoupled attention weights from original unet
        if copy_attn_weights:
            state_dict = self.unet.state_dict()
            for key in state_dict.keys():
                if "_mv" in key:
                    compatible_key = key.replace("_mv", "").replace("processor.", "")
                elif "_ref" in key:
                    compatible_key = key.replace("_ref", "").replace("processor.", "")
                else:
                    compatible_key = key

                is_zero_init_key = any([k in key for k in zero_init_module_keys])
                if is_zero_init_key:
                    state_dict[key] = torch.zeros_like(state_dict[compatible_key])
                else:
                    state_dict[key] = state_dict[compatible_key].clone()
            self.unet.load_state_dict(state_dict)

    def _load_custom_adapter(self, state_dict):
        """Load MV-Adapter weights into unet and condition encoder."""
        self.unet.load_state_dict(state_dict, strict=False)
        self.cond_encoder.load_state_dict(state_dict, strict=False)
        
        # Ensure cond_encoder is on the correct device after loading weights
        device = self.unet.device
        dtype = self.unet.dtype
        self.cond_encoder.to(device=device, dtype=dtype)

    def _save_custom_adapter(
        self,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        """Save MV-Adapter weights."""
        def include_fn(k):
            is_included = False
            if include_keys is not None:
                is_included = is_included or any([key in k for key in include_keys])
            if exclude_keys is not None:
                is_included = is_included and not any([key in k for key in exclude_keys])
            return is_included

        state_dict = {k: v for k, v in self.unet.state_dict().items() if include_fn(k)}
        state_dict.update(self.cond_encoder.state_dict())
        return state_dict
