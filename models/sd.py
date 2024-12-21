from diffusers import StableDiffusionPipeline
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
        super().__init__()
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

        pipe = StableDiffusionPipeline.from_pretrained(config.unet_id)
        self.unet = pipe.unet
        del pipe

        if self._gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # self.unet.requires_grad_(False)

    def forward(self, x, timestep, z_latents, **kwargs):
        z_latents = self.proj(z_latents)
        model_pred = self.unet(x, timestep, z_latents).sample
        return model_pred
