from __future__ import annotations

import copy
import gc
import logging
import os
import types
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
    load_pipeline_config,
    prepare_conditioning,
    seed_everething,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

logger = logging.getLogger("ltx-api")


class LTXEngine:
    """Loads LTX-Video 2B once and reuses it for text-to-video and image-to-video jobs."""

    def __init__(
        self,
        pipeline_config_path: Path,
        max_gpu_memory_gb: float = 0.0,
        cpu_offload: bool = True,
    ) -> None:
        self.pipeline_config_path = Path(pipeline_config_path)
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.cpu_offload = cpu_offload and torch.cuda.is_available()
        self.raw_pipeline_config: dict[str, Any] = load_pipeline_config(
            str(self.pipeline_config_path)
        )
        self.device = get_device()
        self.pipeline = None
        self._skip_layer_strategy: SkipLayerStrategy | None = None
        self._ready = False
        self._accelerate_offload = False

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
        load_device = "cpu" if self.cpu_offload else self.device
        try:
            self.pipeline = create_ltx_video_pipeline(
                ckpt_path=str(ckpt_path),
                precision=precision,
                text_encoder_model_name_or_path=pipeline_config[
                    "text_encoder_model_name_or_path"
                ],
                sampler=pipeline_config.get("sampler"),
                device=load_device,
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
                    raise ValueError(
                        "spatial upscaler weights are required for multi-scale"
                    )
                upsampler_device = load_device
                latent_upsampler = create_latent_upsampler(
                    str(spatial_path), upsampler_device
                )
                self.pipeline = LTXMultiScalePipeline(
                    self.pipeline, latent_upsampler=latent_upsampler
                )
            if self.cpu_offload:
                self._enable_cpu_offload()
        except Exception:
            # Loading can die partway through (e.g. CUDA OOM while moving the
            # transformer/VAE/text-encoder onto the GPU). If we leave the
            # half-built pipeline sitting in self.pipeline, _ready stays
            # False, and every subsequent request calls load() again on top
            # of the still-allocated CUDA memory from this failed attempt —
            # each retry leaks more VRAM until the process is restarted.
            # Tear everything down here so the next attempt starts clean.
            logger.exception("LTX-Video pipeline failed to load, freeing partial state")
            self.pipeline = None
            self._skip_layer_strategy = None
            self._ready = False
            self._accelerate_offload = False
            self._free_cuda()
            raise

        stg_mode = pipeline_config.get("stg_mode", "attention_values")
        self._skip_layer_strategy = self._stg_strategy(stg_mode)
        self._ready = True
        logger.info(
            "LTX-Video pipeline ready on %s (cpu_offload=%s)%s",
            self.device,
            self.cpu_offload,
            f" {self._vram_log()}" if torch.cuda.is_available() else "",
        )

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
        return self.generate(
            prompt=prompt,
            output_path=output_path,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            negative_prompt=negative_prompt,
            image_path=image_path,
            image_cond_noise_scale=image_cond_noise_scale,
        )

    def generate(
        self,
        *,
        prompt: str,
        output_path: str | Path,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        seed: int,
        negative_prompt: str,
        image_path: str | Path | None = None,
        image_cond_noise_scale: float = 0.15,
        single_scale: bool = False,
    ) -> Path:
        if not self._ready:
            self.load()

        pipeline_config = copy.deepcopy(self.raw_pipeline_config)
        pipeline_config.pop("stg_mode", None)
        pipe = self.pipeline
        if single_scale:
            # Server may keep the multi-scale pipeline resident (~11GB). Park
            # unused weights before this small i2v pass so we do not need a
            # different LTX_PIPELINE_CONFIG on the host.
            self.prepare_low_vram()
            pipe = self._inner_pipeline()
            first_pass = pipeline_config.pop("first_pass", None) or {}
            pipeline_config.pop("second_pass", None)
            pipeline_config.pop("pipeline_type", None)
            pipeline_config.pop("downscale_factor", None)
            pipeline_config.pop("spatial_upscaler_model_path", None)
            pipeline_config.update(first_pass)

        seed_everething(seed)
        # Keep the pipeline's T5→CPU→transformer hop even with accelerate, so
        # both models are not resident on GPU at once. Hooks still manage idle.
        offload_to_cpu = self.cpu_offload or get_total_gpu_memory() < 30

        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
        padding = calculate_padding(height, width, height_padded, width_padded)

        try:
            if self.cpu_offload or single_scale:
                self._free_cuda()
                self._place_vae_on_gpu()
            conditioning_items = None
            if image_path is not None:
                conditioning_items = prepare_conditioning(
                    conditioning_media_paths=[str(image_path)],
                    conditioning_strengths=[1.0],
                    conditioning_start_frames=[0],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    padding=padding,
                    pipeline=pipe,
                )

            generator = torch.Generator(device=self.device).manual_seed(seed)
            images = pipe(
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
            if self.cpu_offload or single_scale:
                self._rest_after_job()
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

    def prepare_low_vram(self) -> None:
        """Move transformer/T5/VAE/upscaler to RAM and drop the CUDA cache."""
        self._rest_on_cpu()
        self._free_cuda()
        logger.info("Low-VRAM rest: %s", self._vram_log() or "cpu")

    def _inner_pipeline(self):
        pipe = self.pipeline
        return getattr(pipe, "video_pipeline", pipe)

    def _enable_cpu_offload(self) -> None:
        pipe = self._inner_pipeline()
        self._rest_on_cpu()
        self._wrap_vae_encode_on_gpu(getattr(pipe, "vae", None))
        self._accelerate_offload = self._try_enable_diffusers_offload()
        if self._accelerate_offload:
            logger.info(
                "CPU offload enabled (accelerate): T5/transformer hooked; "
                "VAE excluded so image-to-video lerp stays on CUDA"
            )
            return
        self._force_cuda_execution_device(pipe)
        self._patch_cpu_clears_cache(getattr(pipe, "text_encoder", None))
        self._patch_cpu_clears_cache(getattr(pipe, "transformer", None))
        logger.info(
            "CPU offload enabled (manual, accelerate unavailable): "
            "T5 idles in RAM; VAE is moved to GPU for each job"
        )

    def _try_enable_diffusers_offload(self) -> bool:
        pipe = self._inner_pipeline()
        if pipe is None or not hasattr(pipe, "enable_model_cpu_offload"):
            return False
        excluded = list(getattr(pipe, "_exclude_from_cpu_offload", None) or [])
        if "vae" not in excluded:
            excluded.append("vae")
        pipe._exclude_from_cpu_offload = excluded
        try:
            pipe.enable_model_cpu_offload(device=self.device)
            return True
        except ImportError as exc:
            logger.warning("%s — falling back to manual CPU offload", exc)
            return False

    def _rest_after_job(self) -> None:
        """Free VAE VRAM. Re-apply accelerate hooks if __call__ stripped them."""
        pipe = self._inner_pipeline()
        if pipe is not None and getattr(pipe, "vae", None) is not None:
            pipe.vae.to("cpu")
        if self._accelerate_offload:
            self._try_enable_diffusers_offload()
            self._free_cuda()
            return
        self._rest_on_cpu()

    def _force_cuda_execution_device(self, pipe) -> None:
        """Make pipeline.__call__ move T5/transformer onto CUDA even when weights idle on CPU."""
        if pipe is None:
            return
        device = torch.device(self.device)
        pipe._ltx_cuda_device = device
        cls = type(pipe)
        if getattr(cls, "_ltx_execution_patched", False):
            return
        orig = cls.__dict__.get("_execution_device")

        def _execution_device(this):
            forced = getattr(this, "_ltx_cuda_device", None)
            if forced is not None:
                return forced
            if isinstance(orig, property) and orig.fget is not None:
                return orig.fget(this)
            return device

        cls._execution_device = property(_execution_device)
        cls._ltx_execution_patched = True

    def _place_vae_on_gpu(self) -> None:
        pipe = self._inner_pipeline()
        if pipe is None:
            return
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            vae.to(self.device)
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "to"):
            try:
                scheduler.to(self.device)
            except Exception:
                pass

    def _wrap_vae_encode_on_gpu(self, vae) -> None:
        if vae is None or getattr(vae, "_ltx_encode_on_gpu", False):
            return
        orig_encode = vae.encode
        device = self.device

        def encode(this, x, *args, **kwargs):
            this.to(device)
            if torch.is_tensor(x) and x.device.type != "cuda":
                x = x.to(device)
            return orig_encode(x, *args, **kwargs)

        vae.encode = types.MethodType(encode, vae)
        vae._ltx_encode_on_gpu = True

    def _rest_on_cpu(self) -> None:
        pipe = self._inner_pipeline()
        if pipe is None:
            return
        for name in ("text_encoder", "transformer", "vae"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.to("cpu")
        upsampler = getattr(self.pipeline, "latent_upsampler", None)
        if upsampler is not None:
            upsampler.to("cpu")
        self._free_cuda()

    @staticmethod
    def _patch_cpu_clears_cache(module) -> None:
        if module is None or getattr(module, "_ltx_cpu_clears_cache", False):
            return

        def cpu(this, *args, **kwargs):
            result = torch.nn.Module.cpu(this, *args, **kwargs)
            LTXEngine._free_cuda()
            return result

        module.cpu = types.MethodType(cpu, module)
        module._ltx_cpu_clears_cache = True

    @staticmethod
    def _vram_log() -> str:
        if not torch.cuda.is_available():
            return ""
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"cuda allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"

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
