import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class HyperSDConfig(PretrainedConfig):
    model_type = "hypersd"

    def __init__(
        self,
        unet_id: str = "benjamin-paine/stable-diffusion-v1-5",
        lora_repo: str = "ByteDance/Hyper-SD",
        lora_ckpt: str = "Hyper-SD15-1step-lora.safetensors",
        load_lora: bool = True,
        learn_sigma: bool = False,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self.lora_repo = lora_repo
        self.lora_ckpt = lora_ckpt
        self.load_lora = load_lora
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing


class HyperSD(PreTrainedModel):
    config_class = HyperSDConfig

    def __init__(
        self,
        config: HyperSDConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in hypersd"
        self._gradient_checkpointing = config._gradient_checkpointing
        self.latent_queries = nn.Parameter(torch.randn(1, 77, 768))
        self.proj_in = nn.Linear(config.latent_embedding_size, 768)
        self.adapter_layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    LlamaConfig(
                        _attn_implementation="sdpa",
                        hidden_size=768,
                        intermediate_size=768 * 4,
                        num_attention_heads=12,
                        use_cache=False,
                    ),
                    layer_idx,
                )
                for layer_idx in range(12)
            ],
        )

        self.norm_out = LlamaRMSNorm(768)
        self.proj_out = nn.Linear(768, 768)

        pipe = StableDiffusionPipeline.from_pretrained(config.unet_id)
        if config.load_lora:
            pipe.load_lora_weights(hf_hub_download(config.lora_repo, config.lora_ckpt))
            pipe.fuse_lora()
        self.unet = pipe.unet
        del pipe

        if self._gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.unet.requires_grad_(False)

    def forward(self, x, timestep, z_latents, **kwargs):
        # if not ((z_latents[0] == 0).all() or (z_latents[-1] == 0).all()):
        #     z_latents = self.mask_drop(z_latents)
        latent_queries = self.latent_queries.repeat(z_latents.shape[0], 1, 1)

        z_latents = self.proj_in(z_latents)
        z_latents = torch.cat([z_latents, latent_queries], dim=1)
        for layer in self.adapter_layers:
            if self._gradient_checkpointing and self.training:
                z_latents = torch.utils.checkpoint.checkpoint(
                    layer, z_latents, None, torch.arange(0, z_latents.shape[1], device=z_latents.device).unsqueeze(0)
                )[0]
            else:
                z_latents = layer(z_latents, position_ids=torch.arange(0, z_latents.shape[1], device=z_latents.device).unsqueeze(0))[0]

        z_latents = self.norm_out(z_latents)
        z_latents = self.proj_out(z_latents)
        # get last self.latent_queries.shape[1] latents
        z_latents = z_latents[:, -self.latent_queries.shape[1] :]
        model_pred = self.unet(x, timestep, z_latents).sample
        return model_pred
