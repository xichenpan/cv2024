from typing import List

import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from models.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from torchvision import transforms as v2
from diffusers import LuminaText2ImgPipeline

from models.lumina_nextdit2d import LuminaNextDiT2DModel


class LLaVAOVConfig(PretrainedConfig):
    model_type = "llavaov"

    def __init__(
        self,
        mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        unet_id: str = "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        latent_embedding_size: int = 1152,
        num_pooled_tokens: int = 64,
        learn_sigma: bool = False,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.unet_id = unet_id
        self.latent_embedding_size = latent_embedding_size
        self.num_pooled_tokens = num_pooled_tokens
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing


class LLaVAOV(PreTrainedModel):
    config_class = LLaVAOVConfig

    def __init__(
        self,
        config: LLaVAOVConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in lumina next t2i in context"
        self._gradient_checkpointing = config._gradient_checkpointing

        pipe = LuminaText2ImgPipeline.from_pretrained(config.unet_id)
        self.transformer = LuminaNextDiT2DModel(**pipe.transformer.config)
        self.transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)
        del pipe
        self.text_encoder = LlavaOnevisionForConditionalGeneration.from_pretrained(config.mllm_id)

        self.text_encoder.language_model.lm_head = nn.Identity()
        self.text_encoder.language_model.norm = nn.Identity()
        self.text_encoder.language_model.model.layers = self.text_encoder.language_model.model.layers[:-1]

        self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)
        self.tokenizer.tokenizer.padding_side = "left"

        config.latent_embedding_size = self.text_encoder.language_model.config.hidden_size
        self.latent_queries = nn.Parameter(torch.randn(1, config.num_pooled_tokens, config.latent_embedding_size))
        self.resize_fn = v2.Resize(384)

        self.diffusion_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.text_config.hidden_size, self.transformer.config.cross_attention_dim),
            nn.GELU(),
            nn.Linear(
                self.transformer.config.cross_attention_dim,
                self.transformer.config.cross_attention_dim,
            ),
        )

        if config._gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

        self.freqs_cis = get_2d_rotary_pos_embed_lumina(
            self.transformer.hidden_size // self.transformer.num_attention_heads,
            384,
            384,
        )

    def tokenize(self, caption, image=None):
        if not isinstance(caption, List):
            caption = [caption]
        if image is not None:
            if not isinstance(image, List):
                image = [image]
            image = [self.resize_fn(img) for img in image]

        # Convert each caption into a conversation format
        conversations = [
            [
                {
                    "role": "user",
                    "content": (
                        [
                            {"type": "text", "text": cap},
                            {"type": "image"},
                        ]
                        if image is not None
                        else [{"type": "text", "text": cap}]
                    ),
                }
            ]
            for cap in caption
        ]

        # Apply chat template to each conversation
        prompts = [self.tokenizer.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]

        # Tokenize the formatted prompts
        text_inputs = self.tokenizer(
            images=image,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if image is not None:
            return text_inputs.input_ids, text_inputs.attention_mask, text_inputs.pixel_values, text_inputs.image_sizes
        else:
            return text_inputs.input_ids, text_inputs.attention_mask

    def get_null_ids_and_mask(self, bsz, length, negative_prompt="", image=None):
        conversations = [
            {
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": negative_prompt},
                        {"type": "image"},
                    ]
                    if image is not None
                    else [{"type": "text", "text": negative_prompt}]
                ),
            }
        ]
        prompts = [self.tokenizer.apply_chat_template(conversations, add_generation_prompt=True)]
        text_inputs = self.tokenizer(
            images=image,
            text=prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=length,
        )
        if image is not None:
            return (
                text_inputs.input_ids.repeat(bsz, 1),
                text_inputs.attention_mask.repeat(bsz, 1),
                text_inputs.pixel_values.repeat(bsz, 1, 1, 1, 1),
                text_inputs.image_sizes.repeat(bsz, 1),
            )
        else:
            return text_inputs.input_ids.repeat(bsz, 1), text_inputs.attention_mask.repeat(bsz, 1)

    def encode_condition(self, text_input_ids, attention_mask, z_latents, image_sizes, **kwargs):
        inputs_embeds = self.text_encoder.get_input_embeddings()(text_input_ids)

        latent_queries = self.latent_queries.repeat(inputs_embeds.shape[0], 1, 1)
        inputs_embeds = torch.cat([inputs_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0], dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

        prompt_embeds = self.text_encoder(
            input_ids=torch.cat([text_input_ids, torch.zeros_like(latent_queries[:, :, 0], dtype=text_input_ids.dtype, device=text_input_ids.device)], dim=1),
            pixel_values=z_latents,
            image_sizes=image_sizes,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).logits
        return prompt_embeds[:, -self.config.num_pooled_tokens :], attention_mask[:, -self.config.num_pooled_tokens :]

    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None, z_latents=None, image_sizes=None, **kwargs):
        if prompt_embeds is None or attention_mask is None:
            text_input_ids = kwargs.get("caption")
            attention_mask = kwargs.get("attn_mask")
            prompt_embeds, attention_mask = self.encode_condition(text_input_ids, attention_mask, z_latents, image_sizes)
        model_pred = self.transformer(
            hidden_states=x,
            timestep=1 - timestep / 1000,
            encoder_hidden_states=self.diffusion_proj(prompt_embeds),
            encoder_mask=attention_mask,
            image_rotary_emb=self.freqs_cis,
            cross_attention_kwargs=dict(),
        ).sample
        return -model_pred.chunk(2, dim=1)[0]
