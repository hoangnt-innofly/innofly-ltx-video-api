from __future__ import annotations

import copy
import gc
import logging
import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ltx_video.inference import (
    calculate_padding,
    create_latent_upsampler,
    create_ltx_video_pipeline,
    get_device,
    get_total_gpu_memory,
    load_media_file,
    load_pipeline_config,
    prepare_conditioning,
    seed_everething,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

logger = logging.getLogger("ltx-api")


class LTXEngine:
    """Loads LTX-Video 2B once and reuses it for Image-to-Video jobs."""

    def __init__(
        self,
        pipeline_config_path: Path,
        max_gpu_memory_gb: float = 0.0,
    ) -> None:
        self.pipeline_config_path = Path(pipeline_config_path)
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.raw_pipeline_config: dict[str, Any] = load_pipeline_config(
            str(self.pipeline_config_path)
        )
        self.device = get_device()
        self.pipeline = None
        self._skip_layer_strategy: SkipLayerStrategy | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        self._apply_memory_cap()
        pipeline_config = self.raw_pipeline_config
        ckpt_name = pipeline_config["checkpoint_path"]
        ckpt_path = self._resolve_weight(ckpt_name)

        spatial_name = pipeline_config.get("spatial_upscaler_model_path")
        spatial_path = self._resolve_weight(spatial_name) if spatial_name else None

        precision = pipeline_config["precision"]
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=str(ckpt_path),
            precision=precision,
            text_encoder_model_name_or_path=pipeline_config[
                "text_encoder_model_name_or_path"
            ],
            sampler=pipeline_config.get("sampler"),
            device=self.device,
            enhance_prompt=False,
            prompt_enhancer_image_caption_model_name_or_path=pipeline_config.get(
                "prompt_enhancer_image_caption_model_name_or_path"
            ),
            prompt_enhancer_llm_model_name_or_path=pipeline_config.get(
                "prompt_enhancer_llm_model_name_or_path"
            ),
        )

        if pipeline_config.get("pipeline_type") == "multi-scale":
            if not spatial_path:
                raise ValueError("spatial upscaler weights are required for multi-scale")
            latent_upsampler = create_latent_upsampler(str(spatial_path), self.pipeline.device)
            self.pipeline = LTXMultiScalePipeline(
                self.pipeline, latent_upsampler=latent_upsampler
            )

        stg_mode = pipeline_config.get("stg_mode", "attention_values")
        self._skip_layer_strategy = self._stg_strategy(stg_mode)
        self._ready = True
        logger.info("LTX-Video pipeline ready on %s", self.device)

    def generate_i2v(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        output_path: str | Path,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        seed: int,
        negative_prompt: str,
        image_cond_noise_scale: float = 0.15,
    ) -> Path:
        if not self._ready:
            self.load()

        pipeline_config = copy.deepcopy(self.raw_pipeline_config)
        pipeline_config.pop("stg_mode", None)

        seed_everething(seed)
        offload_to_cpu = get_total_gpu_memory() < 30

        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
        padding = calculate_padding(height, width, height_padded, width_padded)

        try:
            conditioning_items = prepare_conditioning(
                conditioning_media_paths=[str(image_path)],
                conditioning_strengths=[1.0],
                conditioning_start_frames=[0],
                height=height,
                width=width,
                num_frames=num_frames,
                padding=padding,
                pipeline=self.pipeline,
            )

            generator = torch.Generator(device=self.device).manual_seed(seed)
            images = self.pipeline(
                **pipeline_config,
                skip_layer_strategy=self._skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames_padded,
                frame_rate=frame_rate,
                prompt=prompt,
                prompt_attention_mask=None,
                negative_prompt=negative_prompt,
                negative_prompt_attention_mask=None,
                media_items=None,
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=image_cond_noise_scale,
                mixed_precision=(pipeline_config.get("precision") == "mixed_precision"),
                offload_to_cpu=offload_to_cpu,
                device=self.device,
                enhance_prompt=False,
            ).images

            pad_left, pad_right, pad_top, pad_bottom = padding
            pad_bottom = -pad_bottom
            pad_right = -pad_right
            if pad_bottom == 0:
                pad_bottom = images.shape[3]
            if pad_right == 0:
                pad_right = images.shape[4]
            images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

            video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
            del images
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(output_path, fps=frame_rate) as writer:
                for frame in video_np:
                    writer.append_data(frame)
            return output_path
        finally:
            self._free_cuda()

    @staticmethod
    def _resolve_weight(name_or_path: str) -> Path:
        from huggingface_hub import hf_hub_download

        path = Path(name_or_path)
        if path.is_file():
            return path
        downloaded = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=path.name,
            repo_type="model",
        )
        return Path(downloaded)

    @staticmethod
    def _stg_strategy(stg_mode: str) -> SkipLayerStrategy:
        mode = stg_mode.lower()
        if mode in {"stg_av", "attention_values"}:
            return SkipLayerStrategy.AttentionValues
        if mode in {"stg_as", "attention_skip"}:
            return SkipLayerStrategy.AttentionSkip
        if mode in {"stg_r", "residual"}:
            return SkipLayerStrategy.Residual
        if mode in {"stg_t", "transformer_block"}:
            return SkipLayerStrategy.TransformerBlock
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    def _apply_memory_cap(self) -> None:
        cap_gb = self.max_gpu_memory_gb
        if cap_gb <= 0 or not torch.cuda.is_available():
            return
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        fraction = min(1.0, cap_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        logger.info(
            "CUDA memory cap: %.2f GiB of %.2f GiB (fraction=%.3f)",
            cap_gb,
            total_gb,
            fraction,
        )

    @staticmethod
    def _free_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
