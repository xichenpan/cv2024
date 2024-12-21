import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class LCMConfig(PretrainedConfig):
    model_type = "lcm"

    def __init__(
        self,
        unet_id: str = "benjamin-paine/stable-diffusion-v1-5",
        lora_ckpt: str = "latent-consistency/lcm-lora-sdv1-5",
        load_lora: bool = True,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self.lora_ckpt = lora_ckpt
        self.load_lora = load_lora
        self._gradient_checkpointing = _gradient_checkpointing


class LCM(PreTrainedModel):
    config_class = LCMConfig

    def __init__(
        self,
        config: LCMConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in lcm"
        self._gradient_checkpointing = config._gradient_checkpointing
        self.proj = nn.Sequential(
            nn.Linear(config.latent_embedding_size, 768 * 4),
            nn.GELU(),
            nn.Linear(768 * 4, 768),
        )
        pipe = StableDiffusionPipeline.from_pretrained(config.unet_id)
        if config.load_lora:
            pipe.load_lora_weights(config.lora_ckpt)
            pipe.fuse_lora()
        self.unet = pipe.unet
        self.unet.enable_freeu(s1=0.0, s2=0.0, b1=1.0, b2=1.0)
        # self.register_buffer('cfg_null', pipe.encode_prompt("", "cpu", 1, False)[0].mean(1, keepdim=True).detach(),
        #                      persistent=False)
        del pipe

        if self._gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.unet.requires_grad_(False)

    def forward(self, x, timestep, z_latents, **kwargs):
        z_latents = self.proj(z_latents)
        # z_latents = z_latents + self.cfg_null
        model_pred = self.unet(x, timestep, z_latents).sample
        return model_pred
