import os

import torch.utils.data.distributed
from huggingface_hub import login

from trainer_utils import find_newest_checkpoint

login(token="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")
USER_NAME = os.popen("whoami").read().strip()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from typing import Optional, Union, List
from typing import Tuple

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler

from models.soda import SODA

from diffusers import UNet2DConditionModel
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class SDConfig(PretrainedConfig):
    model_type = "sd"

    def __init__(
            self,
            unet_id: str = "benjamin-paine/stable-diffusion-v1-5",
            _gradient_checkpointing: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.unet_id = unet_id
        self._gradient_checkpointing = _gradient_checkpointing


class SD(PreTrainedModel):
    config_class = SDConfig

    def __init__(
            self,
            config: SDConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in sd"
        self._gradient_checkpointing = config._gradient_checkpointing
        self.proj = nn.Sequential(
            nn.Linear(config.latent_embedding_size, 768 * 4),
            nn.GELU(),
            nn.Linear(768 * 4, 768),
        )

        self.unet = UNet2DConditionModel.from_pretrained(config.unet_id, subfolder="unet")

        if self._gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(self, x, timestep, z_latents):
        z_latents = self.proj(z_latents)
        model_pred = self.unet(x, timestep, z_latents).sample
        return model_pred


class SigLIPPipeline(DiffusionPipeline):
    def __init__(self, vae, unet, scheduler):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            features: torch.Tensor,
            guidance_scale: float = 4.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 50,
            output_type: Optional[str] = "pil",
            need_preprocess: bool = True,
            return_dict: bool = True,
            image_size: int = 512,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = features.shape[0]
        latent_size = image_size // 8

        latents = randn_tensor(
            shape=(batch_size, self.vae.latent_channels, latent_size, latent_size),
            generator=generator,
            device=features.device,
            dtype=features.dtype,
        )

        features_null = torch.zeros_like(features, device=features.device)
        features_input = torch.cat([features_null, features], 0) if guidance_scale > 1 else features

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.unet(
                latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                features=features_input
            )

            # perform guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, eta=1.0).prev_sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)


if __name__ == "__main__":
    resume_from_checkpoint = f'/fsx-project/{USER_NAME}/output/diff_sd_siglip_alltokens_1e4_trainunet_81'
    resume_from_checkpoint = find_newest_checkpoint(resume_from_checkpoint)
    soda = SODA.from_pretrained(resume_from_checkpoint, unet_id="benjamin-paine/stable-diffusion-v1-5")
    num_tokens = soda.config.num_pooled_tokens
    training_steps = resume_from_checkpoint.split("/")[-1].split("-")[-1]
    sd = soda.transformer
    sd.push_to_hub("umd-vt-nyu/siglip-sd1.5_all", private=True)

    # pipeline = SigLIPPipeline(
    #     vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae"),
    #     scheduler=DDIMScheduler.from_pretrained("benjamin-paine/stable-diffusion-v1-5", subfolder="scheduler"),
    #     unet=SD.from_pretrained("umd-vt-nyu/siglip-sd1.5"),
    # )
