from typing import List

import torch
from diffusers import LuminaText2ImgPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from models.lumina_nextdit2d import LuminaNextDiT2DModel


class LuminaNextT2IConfig(PretrainedConfig):
    model_type = "luminanext"

    def __init__(
        self,
        unet_id: str = "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        _gradient_checkpointing: bool = False,
        learn_sigma: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self._gradient_checkpointing = _gradient_checkpointing
        self.learn_sigma = learn_sigma


class LuminaNextT2I(PreTrainedModel):
    config_class = LuminaNextT2IConfig

    def __init__(
        self,
        config: LuminaNextT2IConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in sd"
        self._gradient_checkpointing = config._gradient_checkpointing

        pipe = LuminaText2ImgPipeline.from_pretrained(config.unet_id)
        self.max_sequence_length = pipe.max_sequence_length
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.transformer = LuminaNextDiT2DModel(**pipe.transformer.config)
        self.transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)
        del pipe

        if self._gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

        self.text_encoder.layers[-1].requires_grad_(False)
        self.text_encoder.norm.requires_grad_(False)

        self.freqs_cis = get_2d_rotary_pos_embed_lumina(
            self.transformer.hidden_size // self.transformer.num_attention_heads,
            384,
            384,
        )

    def tokenize(self, caption, image=None):
        if not isinstance(caption, List):
            caption = [caption]

        text_inputs = self.tokenizer(
            caption,
            pad_to_multiple_of=8,
            max_length=self.max_sequence_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def get_null_ids_and_mask(self, bsz, seq_len, negative_prompt):
        caption = [negative_prompt] * bsz
        text_inputs = self.tokenizer(
            caption,
            pad_to_multiple_of=8,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def encode_condition(self, z_latents, text_input_ids, attention_mask, **kwargs):
        assert text_input_ids is not None and attention_mask is not None, "text_input_ids and attention_mask are required"
        inputs_embeds = self.text_encoder.embed_tokens(text_input_ids)
        if z_latents is None:
            pass
        else:
            z_latents = self.proj(z_latents)
            z_latents = z_latents.reshape(z_latents.shape[0], -1, z_latents.shape[-1])
            inputs_embeds = torch.cat([z_latents, inputs_embeds], dim=1)
            if attention_mask.shape[1] != inputs_embeds.shape[1]:
                attention_mask = torch.cat([torch.ones_like(z_latents[:, :, 0]), attention_mask], dim=1)

        prompt_embeds = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        return prompt_embeds, attention_mask

    def forward(self, x, timestep, z_latents=None, prompt_embeds=None, attention_mask=None, **kwargs):
        if prompt_embeds is None or attention_mask is None:
            text_input_ids = kwargs.get("caption")
            attention_mask = kwargs.get("attn_mask")
            prompt_embeds, attention_mask = self.encode_condition(z_latents, text_input_ids, attention_mask)
        model_pred = self.transformer(
            hidden_states=x,
            timestep=1 - timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=attention_mask,
            image_rotary_emb=self.freqs_cis,
            cross_attention_kwargs=dict(),
        ).sample
        return -model_pred.chunk(2, dim=1)[0]
