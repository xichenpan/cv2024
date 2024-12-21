from typing import List

import torch
from diffusers import LuminaText2ImgPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from models.lumina_nextdit2d import LuminaNextDiT2DModel


class LuminaNextT2IInContextMSEConfig(PretrainedConfig):

    model_type = "luminanext_icl_mse"

    def __init__(
        self,
        unet_id: str = "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        latent_embedding_size: int = 768,
        num_pooled_tokens: int = 1,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self.latent_embedding_size = latent_embedding_size
        self.num_pooled_tokens = num_pooled_tokens
        self._gradient_checkpointing = _gradient_checkpointing


class LuminaNextT2IInContextMSE(PreTrainedModel):
    config_class = LuminaNextT2IInContextMSEConfig

    def __init__(
        self,
        config: LuminaNextT2IInContextMSEConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing

        pipe = LuminaText2ImgPipeline.from_pretrained(config.unet_id)
        self.max_sequence_length = pipe.max_sequence_length
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        del pipe

        self.proj = nn.Sequential(
            nn.Linear(config.latent_embedding_size, self.text_encoder.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, config.latent_embedding_size),
            nn.GELU(),
            nn.Linear(config.latent_embedding_size, config.latent_embedding_size),
        )
        self.latent_queries = nn.Parameter(torch.randn(1, config.num_pooled_tokens, self.text_encoder.config.hidden_size))

        if self._gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable({"use_reentrant": False})

        self.text_encoder.layers[-1].requires_grad_(False)
        self.text_encoder.norm.requires_grad_(False)

    def tokenize(self, caption):
        if not isinstance(caption, List):
            caption = [caption]

        caption = [" " + cap if cap.startswith("<i>") else cap for cap in caption]

        text_inputs = self.tokenizer(
            caption,
            pad_to_multiple_of=8,
            max_length=self.max_sequence_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def get_null_ids_and_mask(self, bsz, length):
        caption = [""] * bsz
        text_inputs = self.tokenizer(
            caption,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding="max_length",
            max_length=length,
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def encode_condition(self, text_input_ids, attention_mask, z_latents, **kwargs):
        if z_latents is not None:
            z_latents = self.proj(z_latents)

        assert text_input_ids is not None and attention_mask is not None, "text_input_ids and attention_mask are required"
        # replace the <i> token with the z_latents one by one
        inputs_embeds = self.text_encoder.embed_tokens(text_input_ids)
        if z_latents is None:
            pass
        else:
            # Original z_latents shape: [batch_size, num_images, num_tokens, dim]
            bsz, num_images, num_tokens, dim = z_latents.shape

            # Reshape to [batch_size, num_images * (num_tokens), dim]
            z_latents = z_latents.reshape(bsz, -1, dim)
            inputs_embeds = torch.cat([z_latents, inputs_embeds], dim=1)

            # Update attention mask to match the new sequence length
            if attention_mask.shape[1] != inputs_embeds.shape[1]:
                attention_mask = torch.cat([torch.ones_like(z_latents[:, :, 0]), attention_mask], dim=1)

        latent_queries = self.latent_queries.repeat(inputs_embeds.shape[0], 1, 1)
        inputs_embeds = torch.cat([inputs_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)

        prompt_embeds = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        return prompt_embeds[:, -self.config.num_pooled_tokens :], attention_mask[:, -self.config.num_pooled_tokens :]

    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None, z_latents=None, **kwargs):
        if prompt_embeds is None or attention_mask is None:
            text_input_ids = kwargs.get("caption")
            attention_mask = kwargs.get("attn_mask")
            prompt_embeds, attention_mask = self.encode_condition(text_input_ids, attention_mask, z_latents)
        return self.out_proj(prompt_embeds)
