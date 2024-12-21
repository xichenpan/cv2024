from typing import Optional

import torch
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from transformers import PretrainedConfig, PreTrainedModel

from models.lumina_nextdit2d import LuminaNextDiT2DModel


class NextDiTCrossAttnConfig(PretrainedConfig):
    model_type = "nextdit-crossattn"

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 1536,
        n_layers: int = 16,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        latent_embedding_size: int = 1152,
        learn_sigma: bool = False,
        qk_norm: bool = True,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm
        self.latent_embedding_size = latent_embedding_size
        self._gradient_checkpointing = _gradient_checkpointing


class NextDiTCrossAttn(PreTrainedModel):
    config_class = NextDiTCrossAttnConfig

    def __init__(
        self,
        config: NextDiTCrossAttnConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in nextdit-crossattn"
        self._gradient_checkpointing = config._gradient_checkpointing

        self.model = LuminaNextDiT2DModel(
            sample_size=config.input_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.dim,
            num_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            learn_sigma=config.learn_sigma,
            qk_norm=config.qk_norm,
            cross_attention_dim=config.latent_embedding_size,
        )

        if self._gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # self.model.requires_grad_(False)

        self.freqs_cis = get_2d_rotary_pos_embed_lumina(
            config.dim // config.n_heads,
            384,
            384,
        )

    def forward(self, x, timestep, z_latents, prompt_embeds=None, attention_mask=None, **kwargs):
        if prompt_embeds is None or attention_mask is None:
            encoder_hidden_states = z_latents
            encoder_mask = torch.ones((z_latents.shape[0], z_latents.shape[1]), device=z_latents.device)
        else:
            encoder_hidden_states = prompt_embeds
            encoder_mask = attention_mask
        model_pred = self.model(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_mask=encoder_mask,
            image_rotary_emb=self.freqs_cis,
            cross_attention_kwargs=dict(),
        ).sample
        return model_pred
