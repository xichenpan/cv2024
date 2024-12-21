from copy import deepcopy
from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms import InterpolationMode, v2
from transformers import AutoModelForDepthEstimation
from transformers import PreTrainedModel, SiglipImageProcessor, CLIPImageProcessor, AutoImageProcessor
import PIL
from models.encoder import EncoderConfig, Encoder
from models.hypersd import HyperSDConfig, HyperSD
from models.lcm import LCMConfig, LCM
from models.lumina_next_t2i import LuminaNextT2IConfig, LuminaNextT2I
from models.nextdit import NextDiTConfig, NextDiT
from models.nextdit_crossattn import NextDiTCrossAttnConfig, NextDiTCrossAttn
from models.nextdit_uncond import NextDiTUncondConfig, NextDiTUncond
from models.scheduler import LCMDenoiseScheduler, FlowMatchEulerDiscreteScheduler
from models.sd import SDConfig, SD
from models.lumina_next_t2i_icl import LuminaNextT2IInContextConfig, LuminaNextT2IInContext
from models.llavaov import LLaVAOVConfig, LLaVAOV
from models.lumina_next_t2i_icl_mse import LuminaNextT2IInContextMSEConfig, LuminaNextT2IInContextMSE
from models.vlb_loss import GaussianDiffusion
from trainer import EMAMixin
from torchvision.transforms import v2
from trainer_utils import ProcessorWrapper
from tqdm import tqdm


class SODAConfig(
    EncoderConfig,
    NextDiTConfig,
    HyperSDConfig,
    LCMConfig,
    SDConfig,
    LuminaNextT2IConfig,
    NextDiTUncondConfig,
    LuminaNextT2IInContextConfig,
    LLaVAOVConfig,
    NextDiTCrossAttnConfig,
    LuminaNextT2IInContextMSEConfig,
):
    model_type = "soda"

    def __init__(
        self,
        encoder_id: str = "google/siglip-so400m-patch14-384",
        diffusion_model: str = "nextdit",
        vae_id: str = "stabilityai/sdxl-vae",
        vae_downsample_f: int = 8,
        noise_scheduler_id: str = "facebook/DiT-XL-2-256",
        scheduler_id: str = "facebook/DiT-XL-2-256",
        _gradient_checkpointing: bool = False,
        loss_type: str = "diff",
        learn_sigma: bool = False,
        num_sample_steps: int = 50,
        start_timestep_idx: int = 0,
        num_grad_steps: int = 1,
        cut_off_sigma: float = 1,
        first_step_ratio: float = -1,
        num_pooled_tokens: int = -1,
        drop_prob: float = 0.1,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        modules_to_freeze: tuple[str] = ("vae", "transformer"),
        modules_to_unfreeze: tuple[str] = (),
        **kwargs,
    ):
        if encoder_id is not None:
            EncoderConfig.__init__(self, **kwargs)
        if diffusion_model == "nextdit-crossattn":
            NextDiTCrossAttnConfig.__init__(self, **kwargs)
        elif diffusion_model == "nextdit":
            NextDiTConfig.__init__(self, **kwargs)
        elif diffusion_model == "nextdit-uncond":
            NextDiTUncondConfig.__init__(self, **kwargs)
        elif diffusion_model == "hypersd":
            HyperSDConfig.__init__(self, **kwargs)
        elif diffusion_model == "lcm":
            LCMConfig.__init__(self, **kwargs)
        elif diffusion_model == "sd":
            SDConfig.__init__(self, **kwargs)
        elif diffusion_model == "luminanext":
            LuminaNextT2IConfig.__init__(self, **kwargs)
        elif diffusion_model == "luminanext-icl":
            LuminaNextT2IInContextConfig.__init__(self, **kwargs)
        elif diffusion_model == "llavaov":
            LLaVAOVConfig.__init__(self, **kwargs)
        elif diffusion_model == "luminanext-icl-mse":
            LuminaNextT2IInContextMSEConfig.__init__(self, **kwargs)
        else:
            raise ValueError(f"Unknown diffusion model {diffusion_model}")

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.encoder_id = encoder_id
        self.diffusion_model = diffusion_model
        self.vae_id = vae_id
        self.vae_downsample_f = vae_downsample_f
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self._gradient_checkpointing = _gradient_checkpointing
        self.loss_type = loss_type
        self.learn_sigma = learn_sigma
        self.num_sample_steps = num_sample_steps
        self.start_timestep_idx = start_timestep_idx
        self.num_grad_steps = num_grad_steps
        self.cut_off_sigma = cut_off_sigma
        self.first_step_ratio = first_step_ratio
        self.num_pooled_tokens = num_pooled_tokens
        self.drop_prob = drop_prob
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_unfreeze = modules_to_unfreeze


class SODA(PreTrainedModel, EMAMixin):
    config_class = SODAConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        EMAMixin.__init__(self, config)
        assert (
            config.start_timestep_idx < config.num_sample_steps
        ), f"start_timestep_idx {config.start_timestep_idx} must be less than num_sample_steps {config.num_sample_steps}"
        self.config = config
        self.start_timestep_idx = config.start_timestep_idx

        if config.encoder_id is not None:
            self.encoder = Encoder(EncoderConfig(**config.to_dict()))
            config.latent_embedding_size = self.encoder.model.config.hidden_size

        if config.encoder_id is None:
            self.processor = None
            self.source_transform = None
            self.source_image_size = None
        else:
            if "siglip" in config.encoder_id:
                self.processor = SiglipImageProcessor.from_pretrained(config.encoder_id)
                self.source_image_size = min(self.processor.size["height"], self.processor.size["width"])
            elif "clip" in config.encoder_id:
                self.processor = CLIPImageProcessor.from_pretrained(config.encoder_id)
                self.source_image_size = self.processor.size["shortest_edge"]
            elif "dino" in config.encoder_id:
                self.processor = AutoImageProcessor.from_pretrained(config.encoder_id)
                self.source_image_size = self.processor.size["shortest_edge"]
            else:
                raise ValueError(f"Unknown model_id: {config.encoder_id}")
            self.source_transform = v2.Compose(
                [
                    v2.Resize(self.source_image_size),
                    v2.CenterCrop(self.source_image_size),
                    ProcessorWrapper(self.processor),
                ]
            )

        if config.diffusion_model == "nextdit-crossattn":
            self.transformer = NextDiTCrossAttn(NextDiTCrossAttnConfig(**config.to_dict()))
        elif config.diffusion_model == "nextdit":
            self.transformer = NextDiT(NextDiTConfig(**config.to_dict()))
        elif config.diffusion_model == "nextdit-uncond":
            self.transformer = NextDiTUncond(NextDiTUncondConfig(**config.to_dict()))
        elif config.diffusion_model == "hypersd":
            self.transformer = HyperSD(HyperSDConfig(**config.to_dict()))
        elif config.diffusion_model == "lcm":
            self.transformer = LCM(LCMConfig(**config.to_dict()))
        elif config.diffusion_model == "sd":
            self.transformer = SD(SDConfig(**config.to_dict()))
        elif config.diffusion_model == "luminanext":
            self.transformer = LuminaNextT2I(LuminaNextT2IConfig(**config.to_dict()))
        elif config.diffusion_model == "luminanext-icl":
            self.transformer = LuminaNextT2IInContext(LuminaNextT2IInContextConfig(**config.to_dict()))
        elif config.diffusion_model == "llavaov":
            self.transformer = LLaVAOV(LLaVAOVConfig(**config.to_dict()))
        elif config.diffusion_model == "luminanext-icl-mse":
            self.transformer = LuminaNextT2IInContextMSE(LuminaNextT2IInContextMSEConfig(**config.to_dict()))
        else:
            raise ValueError(f"Unknown diffusion model {config.diffusion_model}")

        self.loss_type = config.loss_type

        if self.loss_type == "mse":
            self.vae = None
        elif "FLUX" in config.vae_id:
            self.vae = AutoencoderKL.from_pretrained(config.vae_id, subfolder="vae")
        else:
            self.vae = AutoencoderKL.from_pretrained(config.vae_id)

        if config.diffusion_model == "lcm":
            self.noise_scheduler = LCMDenoiseScheduler.from_pretrained(config.noise_scheduler_id, subfolder="scheduler")
        elif self.loss_type in ["flow", "flow_sample", "flow_sample_depth"]:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(cut_off_sigma=config.cut_off_sigma)
        elif self.loss_type in ["diff", "sds"]:
            self.noise_scheduler = DDPMScheduler.from_pretrained(config.noise_scheduler_id, subfolder="scheduler")
        elif self.loss_type in ["sample", "sample_depth", "dpo"]:
            self.noise_scheduler = DDIMScheduler.from_pretrained(config.noise_scheduler_id, subfolder="scheduler")
            self.noise_scheduler.set_timesteps(config.num_sample_steps)
        elif self.loss_type == "mse":
            pass
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        if config.diffusion_model == "lcm":
            self.scheduler = LCMScheduler.from_pretrained(config.scheduler_id, subfolder="scheduler")
        elif self.loss_type in ["flow", "flow_sample", "flow_sample_depth"]:
            self.scheduler = FlowMatchEulerDiscreteScheduler(cut_off_sigma=config.cut_off_sigma)
        else:
            self.scheduler = DDIMScheduler.from_pretrained(config.scheduler_id, subfolder="scheduler")

        if self.loss_type in ["sample_depth", "flow_sample_depth"]:
            self.depth_processor = v2.Compose(
                [
                    v2.Resize((518, 518), interpolation=InterpolationMode.BICUBIC),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_model.requires_grad_(False)
            if config._gradient_checkpointing:
                self.depth_model.gradient_checkpointing_enable()
        if config.learn_sigma:
            self.vlb_loss = GaussianDiffusion(
                alphas=self.noise_scheduler.alphas,
                alphas_cumprod=self.noise_scheduler.alphas_cumprod,
                betas=self.noise_scheduler.betas,
            )

        if self.vae is not None:
            self.vae.eval()
            self.vae.requires_grad_(False)

        if hasattr(self, "encoder") and "encoder" in config.modules_to_freeze:
            self.encoder.requires_grad_(False)
        if "transformer" in config.modules_to_freeze:
            self.transformer.requires_grad_(False)
            if hasattr(self.transformer, "proj") and "proj" not in config.modules_to_freeze:
                self.transformer.proj.requires_grad_(True)

        for module_name in config.modules_to_freeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(False)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(False)

        for module_name in config.modules_to_unfreeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(True)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(True)

    def init_copy(self):
        # after load state dict, need to copy params
        if self.config.use_ema:
            self.ema_encoder = type(self.encoder)(self.encoder.config)
            self.ema_encoder.load_state_dict(deepcopy(self.encoder.state_dict()))
            self.encoder.model.gradient_checkpointing_disable()
            self.ema_encoder.model.gradient_checkpointing_disable()
            self.ema_encoder.requires_grad_(False)
        else:
            self.ema_encoder = None

        if self.config.loss_type == "dpo":
            self.ref_encoder = type(self.encoder)(self.encoder.config)
            self.ref_encoder.load_state_dict(deepcopy(self.encoder.state_dict()))
            self.ref_encoder.requires_grad_(False)

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1

        return latents * mask

    def tokenize(self, caption, image=None):
        if hasattr(self, "encoder") and hasattr(self.encoder, "tokenizer"):
            return self.encoder.tokenize(caption)
        elif hasattr(self.transformer, "tokenizer"):
            return self.transformer.tokenize(caption, image)
        else:
            raise ValueError("No tokenizer found")

    def forward(self, x_target, x_source=None, caption=None, attn_mask=None, **kwargs):
        if self.loss_type == "mse":
            bsz, num_images = x_source.shape[:2]
            z_latents, _ = self.encoder(x_source.flatten(0, 1), caption=None, num_pooled_tokens=self.config.num_pooled_tokens)
            z_latents = z_latents.reshape(bsz, num_images, z_latents.shape[-2], z_latents.shape[-1])
            gt_idx = kwargs.get("gt_idx")
            target = z_latents[torch.arange(bsz), gt_idx]

            model_pred = self.transformer(
                x=None,
                timestep=None,
                z_latents=z_latents,
                caption=caption,
                attn_mask=attn_mask,
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            return {"loss": loss}

        if isinstance(self.vae, AutoencoderKL):
            latents = self.vae.encode(x_target).latent_dist.sample()
            if self.vae.config.shift_factor is not None:
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor
        else:
            latents = self.vae.encode(x_target)

        bsz = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        if x_source is None:
            z_latents = None
        elif self.config.diffusion_model == "llavaov":
            z_latents = x_source
        else:
            # Get valid indices from attention mask
            x_source_mask = attn_mask[:, : -caption.shape[1]]  # batch x seq
            x_source_mask = x_source_mask.view(
                x_source_mask.shape[0],
                -1,
                # self.config.num_pooled_tokens if self.config.num_pooled_tokens > 0 else 729,
                self.config.num_pooled_tokens + 2 if self.config.num_pooled_tokens > 0 else 731,
            )
            x_source_mask = x_source_mask.any(dim=-1)  # batch x (seq/num_pooled_tokens)

            # Get valid indices and reshape x_source in-place
            valid_indices = x_source_mask.nonzero()  # [N, 2] tensor of [batch_idx, seq_idx]
            x_source = x_source[valid_indices[:, 0], valid_indices[:, 1]]

            # Encode valid images
            z_latents, _ = self.encoder(x_source, caption=None, num_pooled_tokens=self.config.num_pooled_tokens)

            # Create output tensor and fill with encoded values
            output_shape = (attn_mask.shape[0], x_source_mask.shape[1], *z_latents.shape[1:])
            z_latents_out = torch.zeros(output_shape, device=z_latents.device, dtype=z_latents.dtype)
            z_latents_out[valid_indices[:, 0], valid_indices[:, 1]] = z_latents

            # Final reshape
            z_latents = z_latents_out.reshape(bsz, -1, z_latents.shape[-2], z_latents.shape[-1])

        if caption is None:
            caption, attn_mask = self.tokenize("").to(device=device, dtype=dtype)
            caption = caption.repeat(bsz, 1)
            attn_mask = attn_mask.repeat(bsz, 1)

        noise = torch.randn_like(latents, device=latents.device)
        # noise += 0.1 * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

        if self.loss_type == "flow" and isinstance(self.transformer, NextDiTUncond):
            u = torch.rand(size=(bsz,), device="cpu")
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            input_latents = (1.0 - sigmas) * latents + sigmas * z_latents
            model_pred = self.transformer(
                x=input_latents,
                timestep=timesteps,
            )

            target = z_latents - latents
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        elif self.loss_type == "flow":
            u = torch.rand(size=(bsz,), device="cpu")
            if self.config.first_step_ratio > 0:
                u = torch.where(torch.rand_like(u) <= self.config.first_step_ratio, torch.zeros_like(u, device=u.device, dtype=u.dtype), u)
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            prompt_embeds, attention_mask = self.transformer.encode_condition(
                text_input_ids=caption,
                attention_mask=attn_mask,
                z_latents=z_latents,
                image_sizes=kwargs.get("image_sizes", None),
            )

            noise_pred = self.transformer(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )

            target = noise - latents
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        elif self.loss_type == "flow_sample":
            guidance_scale = 7.5

            sigmas = np.linspace(1.0, 1 / self.config.num_sample_steps, self.config.num_sample_steps)
            self.noise_scheduler.set_timesteps(self.config.num_sample_steps, sigmas=sigmas)

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            if caption is None:
                z_latents_input = torch.cat([z_latents_null, z_latents], 0)
                caption_input, attn_mask_input = None, None
            else:
                caption_null = torch.zeros_like(caption, device=caption.device)
                caption_null[:, :1] = 2
                caption_null[:, 1:2] = 1
                attn_mask_null = torch.zeros_like(attn_mask, device=attn_mask.device)
                attn_mask_null[:, :2] = 1
                z_latents_input = torch.cat([z_latents_null, z_latents], 0)
                caption_input = torch.cat([caption_null, caption], 0)
                attn_mask_input = torch.cat([attn_mask_null, attn_mask], 0)

            noisy_latents = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

            prompt_embeds, attention_mask = self.transformer.encode_condition(
                text_input_ids=caption_input,
                attention_mask=attn_mask_input,
                z_latents=z_latents_input,
                image_sizes=kwargs.get("image_sizes", None),
            )

            for i, t in enumerate(self.noise_scheduler.timesteps):
                t_tensor = t.unsqueeze(0).expand(latents.shape[0]).to(latents.device, torch.long)
                latent_model_input = latents.repeat(len(z_latents_input) // len(z_latents), 1, 1, 1)
                t_tensor_input = t_tensor.repeat(len(z_latents_input) // len(z_latents))

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timestep=t_tensor_input,
                    z_latents=None,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                )

                # perform guidance
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous image: x_t -> x_t-1
                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents).prev_sample
            loss = F.mse_loss(noisy_latents.float(), latents.float(), reduction="mean")

        elif self.loss_type == "flow_sample_depth":
            guidance_scale = 0.0

            sigmas = np.linspace(1.0, 1 / self.config.num_sample_steps, self.config.num_sample_steps)
            image_seq_len = latents.shape[1]
            self.noise_scheduler.set_timesteps(self.config.num_sample_steps, sigmas=sigmas)

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            z_latents_input = torch.cat([z_latents_null, z_latents], 0) if guidance_scale > 1 else z_latents
            noisy_latents = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

            for i, t in enumerate(self.noise_scheduler.timesteps):
                t_tensor = t.unsqueeze(0).expand(latents.shape[0]).to(latents.device, torch.long)
                latent_model_input = torch.cat([noisy_latents] * 2) if guidance_scale > 1 else noisy_latents
                t_tensor_input = t_tensor.repeat(2) if guidance_scale > 1 else t_tensor

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timestep=t_tensor_input,
                    z_latents=z_latents_input,
                    caption=caption,
                    attn_mask=attn_mask,
                )

                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous image: x_t -> x_t-1
                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents).prev_sample
            image_size = self.config.input_size * self.config.vae_downsample_f
            noisy_depth = self.depth_model(pixel_values=self.depth_processor(self.decode_latents(noisy_latents, return_tensor=True))).predicted_depth
            noisy_depth = torch.nn.functional.interpolate(
                noisy_depth.unsqueeze(1),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            )
            target_depth = self.depth_model(pixel_values=self.depth_processor(x_target / 2 + 0.5)).predicted_depth
            target_depth = torch.nn.functional.interpolate(
                target_depth.unsqueeze(1),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            )
            loss = F.mse_loss(noisy_depth.float(), target_depth.float(), reduction="mean")

        elif self.loss_type == "diff":
            guidance_scale = 0.0

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            z_latents_input = torch.cat([z_latents_null, z_latents], 0) if guidance_scale > 1 else z_latents

            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            latent_model_input = torch.cat([noisy_latents] * 2) if guidance_scale > 1 else noisy_latents
            timesteps_input = timesteps.repeat(2) if guidance_scale > 1 else timesteps
            noise_pred = self.transformer(
                x=latent_model_input,
                timestep=timesteps_input,
                z_latents=z_latents_input,
                caption=caption,
                attn_mask=attn_mask,
            )
            # learned sigma
            if self.config.learn_sigma:
                noise_pred, var_pred = torch.split(noise_pred, self.config.in_channels, dim=1)

            # perform guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                if self.config.learn_sigma:
                    var_pred_uncond, var_pred = var_pred.chunk(2)
                    var_pred = var_pred_uncond + guidance_scale * (var_pred - var_pred_uncond)

            mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            if self.config.learn_sigma:
                frozen_out = torch.cat([noise_pred.detach(), var_pred], dim=1)
                vlb_loss = self.vlb_loss(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=noisy_latents,
                    t=timesteps,
                    clip_denoised=False,
                )["output"].mean()
            else:
                vlb_loss = None

            loss = (mse_loss + vlb_loss) if vlb_loss is not None else mse_loss

        elif self.loss_type == "cm":
            guidance_scale = 0.0

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            z_latents_input = torch.cat([z_latents_null, z_latents], 0) if guidance_scale > 1 else z_latents

            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            latent_model_input = torch.cat([noisy_latents] * 2) if guidance_scale > 1 else noisy_latents
            timesteps_input = timesteps.repeat(2) if guidance_scale > 1 else timesteps
            noise_pred = self.transformer(
                x=latent_model_input,
                timestep=timesteps_input,
                z_latents=z_latents_input,
                caption=caption,
                attn_mask=attn_mask,
            )
            # learned sigma
            if self.config.learn_sigma:
                noise_pred, var_pred = torch.split(noise_pred, self.config.in_channels, dim=1)

            # perform guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                if self.config.learn_sigma:
                    var_pred_uncond, var_pred = var_pred.chunk(2)
                    var_pred = var_pred_uncond + guidance_scale * (var_pred - var_pred_uncond)

            # mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            mse_loss = F.mse_loss(self.noise_scheduler.step(noise_pred, timesteps, noisy_latents).denoised.float(), latents.float(), reduction="mean")

            if self.config.learn_sigma:
                frozen_out = torch.cat([noise_pred.detach(), var_pred], dim=1)
                vlb_loss = self.vlb_loss(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=noisy_latents,
                    t=timesteps,
                    clip_denoised=False,
                )["output"].mean()
            else:
                vlb_loss = None

            loss = (mse_loss + vlb_loss) if vlb_loss is not None else mse_loss

        elif self.loss_type == "sample":
            guidance_scale = 7.5

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            z_latents_input = torch.cat([z_latents_null, z_latents], 0) if guidance_scale > 1 else z_latents
            noisy_latents = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

            for i, t in enumerate(self.noise_scheduler.timesteps):
                t_tensor = t.unsqueeze(0).expand(latents.shape[0]).to(latents.device, torch.long)
                latent_model_input = torch.cat([noisy_latents] * 2) if guidance_scale > 1 else noisy_latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                t_tensor_input = t_tensor.repeat(2) if guidance_scale > 1 else t_tensor

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timestep=t_tensor_input,
                    z_latents=z_latents_input,
                    caption=caption,
                    attn_mask=attn_mask,
                )

                # learned sigma
                if self.config.learn_sigma:
                    noise_pred, var_pred = torch.split(noise_pred, self.config.in_channels, dim=1)

                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous image: x_t -> x_t-1
                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents).prev_sample
            loss = F.mse_loss(noisy_latents.float(), latents.float(), reduction="mean")

        elif self.loss_type == "sample_depth":
            guidance_scale = 7.5

            z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
            z_latents_input = torch.cat([z_latents_null, z_latents], 0) if guidance_scale > 1 else z_latents
            noisy_latents = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

            for i, t in enumerate(self.noise_scheduler.timesteps):
                t_tensor = t.unsqueeze(0).expand(latents.shape[0]).to(latents.device, torch.long)
                latent_model_input = torch.cat([noisy_latents] * 2) if guidance_scale > 1 else noisy_latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                t_tensor_input = t_tensor.repeat(2) if guidance_scale > 1 else t_tensor

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timestep=t_tensor_input,
                    z_latents=z_latents_input,
                    caption=caption,
                    attn_mask=attn_mask,
                )

                # learned sigma
                if self.config.learn_sigma:
                    noise_pred, var_pred = torch.split(noise_pred, self.config.in_channels, dim=1)

                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous image: x_t -> x_t-1
                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents).prev_sample
            image_size = self.config.input_size * self.config.vae_downsample_f
            noisy_depth = self.depth_model(pixel_values=self.depth_processor(self.decode_latents(noisy_latents, return_tensor=True))).predicted_depth
            noisy_depth = torch.nn.functional.interpolate(
                noisy_depth.unsqueeze(1),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            )
            target_depth = self.depth_model(pixel_values=self.depth_processor(x_target / 2 + 0.5)).predicted_depth
            target_depth = torch.nn.functional.interpolate(
                target_depth.unsqueeze(1),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            )
            loss = F.mse_loss(noisy_depth.float(), target_depth.float(), reduction="mean")

        elif self.loss_type == "sds":
            assert isinstance(self.transformer, NextDiTUncond), "Only NextDiTUncond supports SDS loss"

            u = torch.rand(size=(bsz,), device="cpu")
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            input_latents = (1.0 - sigmas) * latents + sigmas * z_latents

            target = z_latents - latents
            with torch.inference_mode():
                model_pred = self.transformer(
                    x=input_latents,
                    timestep=timesteps,
                )
            grad = model_pred - target
            grad = torch.nan_to_num(grad.detach(), 0.0, 0.0, 0.0)
            loss = (grad.clone() * z_latents).mean()
            del grad

        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        return {"loss": loss}

    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if isinstance(self.vae, AutoencoderKL):
            latents = latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            samples = self.vae.decode(latents).sample
        else:
            samples = self.vae.decode(latents)
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def sample_images(
        self,
        x_source=None,
        caption=None,
        attn_mask=None,
        guidance_scale: float = 3.0,
        text_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        negative_prompt="",
        enable_progress_bar=False,
        **kwargs,
    ):
        if isinstance(self.transformer, NextDiTUncond):
            return self.sample_images_uncond(x_source, num_inference_steps, return_tensor, **kwargs)
        elif isinstance(self.transformer, LuminaNextT2IInContextMSE):
            return self.sample_images_mse(x_source, caption, attn_mask, **kwargs)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        bsz = caption.shape[0] if caption is not None else 1

        if self.config.diffusion_model == "llavaov":
            if x_source is None:
                z_latents = None
                z_latents_input = None
                image_sizes_input = None
                caption_null, attn_mask_null = self.transformer.get_null_ids_and_mask(caption.shape[0], caption.shape[1], negative_prompt)
                caption_null = caption_null.to(device, caption.dtype)
                attn_mask_null = attn_mask_null.to(device, attn_mask.dtype)
                caption_input = torch.cat([caption_null, caption_null, caption], 0)
                attn_mask_input = torch.cat([attn_mask_null, attn_mask_null, attn_mask], 0)
            else:
                z_latents = x_source
                image_sizes = kwargs.get("image_sizes")
                width, height = image_sizes[0]
                caption_null, attn_mask_null, z_latents_null, image_sizes_null = self.transformer.get_null_ids_and_mask(
                    caption.shape[0], caption.shape[1], negative_prompt, image=PIL.Image.new("RGB", (height, width))
                )
                caption_null = caption_null.to(device, caption.dtype)
                attn_mask_null = attn_mask_null.to(device, attn_mask.dtype)
                z_latents_null = z_latents_null.to(device, z_latents.dtype)
                image_sizes_null = image_sizes_null.to(device, image_sizes.dtype)
                caption_input = torch.cat([caption_null, caption_null, caption], 0)
                attn_mask_input = torch.cat([attn_mask_null, attn_mask_null, attn_mask], 0)
                z_latents_input = torch.cat([z_latents_null, z_latents, z_latents], 0)
                image_sizes_input = torch.cat([image_sizes_null, image_sizes, image_sizes], 0)

        else:
            image_sizes_input = None
            if x_source is None:
                z_latents = None
                z_latents_input = None
            else:
                if x_source.ndim > 4:
                    x_source = x_source.squeeze(0)
                num_images, _, height, width = x_source.shape
                x_source_null = self.source_transform(PIL.Image.new("RGB", (height, width))).unsqueeze(0).repeat(num_images, 1, 1, 1).to(device, dtype)
                z_latents = self.encoder(x_source, caption=None, num_pooled_tokens=self.config.num_pooled_tokens)[0]
                z_latents = z_latents.reshape(bsz, -1, z_latents.shape[-2], z_latents.shape[-1])
                z_latents_null = self.encoder(x_source_null, caption=None, num_pooled_tokens=self.config.num_pooled_tokens)[0]
                z_latents_null = z_latents_null.reshape(bsz, -1, z_latents_null.shape[-2], z_latents_null.shape[-1])
                z_latents_input = torch.cat([z_latents_null, z_latents, z_latents], 0)

            if caption is None:
                caption, attn_mask = self.tokenize("").to(device, dtype)
            caption_null, attn_mask_null = self.transformer.get_null_ids_and_mask(caption.shape[0], caption.shape[1], negative_prompt)
            caption_null = caption_null.to(device, caption.dtype)
            attn_mask_null = attn_mask_null.to(device, attn_mask.dtype)
            caption_input = torch.cat([caption_null, caption_null, caption], 0)
            attn_mask_input = torch.cat([attn_mask_null, attn_mask_null, attn_mask], 0)

        batch_size = z_latents.shape[0] if z_latents is not None else caption.shape[0]
        latent_size = self.config.input_size
        latent_channels = self.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        # Repeat z_latents and conditions for each image per prompt
        caption_input = caption_input.repeat_interleave(num_images_per_prompt, dim=0)
        attn_mask_input = attn_mask_input.repeat_interleave(num_images_per_prompt, dim=0)
        z_latents_input = z_latents_input.repeat_interleave(num_images_per_prompt, dim=0) if z_latents_input is not None else None
        image_sizes_input = image_sizes_input.repeat_interleave(num_images_per_prompt, dim=0) if image_sizes_input is not None else None

        prompt_embeds, attention_mask = self.transformer.encode_condition(
            text_input_ids=caption_input,
            attention_mask=attn_mask_input,
            z_latents=z_latents_input,
            image_sizes=image_sizes_input,
        )

        # Convert to float32 before saving
        for t in tqdm(self.scheduler.timesteps, desc="Sampling images", disable=not enable_progress_bar):
            latent_model_input = latents.repeat(len(caption_input) // len(latents), 1, 1, 1)
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.transformer(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )

            # learned sigma
            if self.config.learn_sigma:
                noise_pred = torch.split(noise_pred, latent_channels, dim=1)[0]

            # perform guidance
            if caption is None:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            else:
                noise_pred_uncond, noise_pred_uncond_text, noise_pred = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond + guidance_scale * (noise_pred_uncond_text - noise_pred_uncond) + text_guidance_scale * (noise_pred - noise_pred_uncond_text)
                )

            # compute previous image: x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, eta=1.0).prev_sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latents(latents, return_tensor=return_tensor)
        return samples

    def sample_images_uncond(
        self,
        x_source=None,
        num_inference_steps: int = 30,
        return_tensor=False,
        **kwargs,
    ):
        latents = self.encoder(x_source, caption=None, num_pooled_tokens=self.config.num_pooled_tokens)[0]

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        for t in self.scheduler.timesteps:
            model_pred = self.transformer(
                x=latents,
                timestep=t.unsqueeze(0).expand(latents.shape[0]).to(latents.device, torch.long),
            )
            latents = self.scheduler.step(model_pred, t, latents).prev_sample

        samples = self.decode_latents(latents, return_tensor=return_tensor)
        return samples

    def sample_images_mse(
        self,
        x_source=None,
        caption=None,
        attn_mask=None,
        **kwargs,
    ):
        gt_idx = kwargs.get("gt_idx")
        bsz, num_images = x_source.shape[:2]
        z_latents, _ = self.encoder(x_source.flatten(0, 1), caption=None, num_pooled_tokens=self.config.num_pooled_tokens)
        z_latents = z_latents.reshape(bsz, num_images, z_latents.shape[-2], z_latents.shape[-1])

        model_pred = self.transformer(
            x=None,
            timestep=None,
            z_latents=z_latents,
            caption=caption,
            attn_mask=attn_mask,
        )
        # Calculate cosine similarity for each sequence position and average
        cos_sims = []
        for i in range(num_images):
            sim = F.cosine_similarity(model_pred, z_latents[:, i], dim=-1)  # [1, 64]
            cos_sims.append(sim.mean(dim=1))  # Average across sequence positions

        cos_sims = torch.stack(cos_sims, dim=1)  # [batch_size, num_images]

        # Create black/white images based on whether predicted index matches gt_idx
        pred_idx = cos_sims.argmax(dim=1)
        correct = pred_idx == gt_idx

        # Create white image for correct predictions, black for incorrect
        image_size = self.config.input_size * self.config.vae_downsample_f
        images = torch.where(
            correct.view(-1, 1, 1, 1),
            torch.ones(bsz, 3, image_size, image_size, device=model_pred.device),
            torch.zeros(bsz, 3, image_size, image_size, device=model_pred.device),
        )

        return images
