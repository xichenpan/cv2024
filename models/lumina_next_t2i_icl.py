from typing import List

import torch
from diffusers import LuminaText2ImgPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from models.lumina_nextdit2d import LuminaNextDiT2DModel


class LuminaNextT2IInContextConfig(PretrainedConfig):
    model_type = "luminanext_icl"

    def __init__(
        self,
        unet_id: str = "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        latent_embedding_size: int = 1152,
        num_pooled_tokens: int = 64,
        learn_sigma: bool = False,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.unet_id = unet_id
        self.latent_embedding_size = latent_embedding_size
        self.num_pooled_tokens = num_pooled_tokens
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing


class CombinedLatentEmbedding(nn.Module):
    def __init__(self, original_embedding_module: nn.Embedding, new_embedding_module: nn.Embedding):
        super().__init__()
        self.original_embedding_module = original_embedding_module
        self.new_embedding_module = new_embedding_module
        self.original_embedding_num_embeddings = original_embedding_module.num_embeddings

    def forward(self, input_ids):
        # Create masks for original and new embeddings
        original_mask = input_ids < self.original_embedding_num_embeddings
        new_mask = ~original_mask

        # Initialize output tensor with same shape and device as would be produced by embedding
        output = torch.zeros((*input_ids.shape, self.original_embedding_module.embedding_dim), dtype=self.original_embedding_module.weight.dtype, device=input_ids.device)

        # Handle original embeddings
        if original_mask.any():
            original_ids = input_ids[original_mask]
            output[original_mask] = self.original_embedding_module(original_ids)

        # Handle new embeddings
        if new_mask.any():
            new_ids = input_ids[new_mask] - self.original_embedding_num_embeddings
            output[new_mask] = self.new_embedding_module(new_ids)

        return output


class LuminaNextT2IInContext(PreTrainedModel):
    config_class = LuminaNextT2IInContextConfig

    def __init__(
        self,
        config: LuminaNextT2IInContextConfig,
    ) -> None:
        super().__init__(config)
        assert config.learn_sigma is False, "learn_sigma is not supported in lumina next t2i in context"
        self._gradient_checkpointing = config._gradient_checkpointing

        pipe = LuminaText2ImgPipeline.from_pretrained(config.unet_id)
        self.max_sequence_length = pipe.max_sequence_length
        self.tokenizer = pipe.tokenizer
        new_tokens = ["<img>", "<\img>", " <i>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<\img>")
        self.text_encoder = pipe.text_encoder

        # Create new embedding for additional tokens
        new_embed_tokens = nn.Embedding(
            len(new_tokens),
            self.text_encoder.embed_tokens.embedding_dim,
            padding_idx=None,
        )
        # Replace the original embedding with CombinedLatentEmbedding
        self.text_encoder.embed_tokens = CombinedLatentEmbedding(self.text_encoder.embed_tokens, new_embed_tokens)

        self.transformer = LuminaNextDiT2DModel(**pipe.transformer.config)
        self.transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)
        del pipe

        self.mm_proj = nn.Sequential(
            nn.Linear(config.latent_embedding_size, self.text_encoder.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
        )

        # self.diffusion_proj = nn.Sequential(
        #     nn.Linear(self.text_encoder.config.hidden_size, self.transformer.config.cross_attention_dim),
        #     nn.GELU(),
        #     nn.Linear(self.transformer.config.cross_attention_dim, self.transformer.config.cross_attention_dim),
        # )

        self.latent_queries = nn.Parameter(torch.randn(1, config.num_pooled_tokens, self.transformer.config.cross_attention_dim))

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

    def get_null_ids_and_mask(self, bsz, length, negative_prompt=""):
        caption = [negative_prompt] * bsz
        text_inputs = self.tokenizer(
            caption,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding="max_length",
            max_length=length,
            truncation=True,
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def encode_condition(self, z_latents, text_input_ids, attention_mask, **kwargs):
        if z_latents is not None:
            z_latents = self.mm_proj(z_latents)

        assert text_input_ids is not None and attention_mask is not None, "text_input_ids and attention_mask are required"
        # get where we insert the <i> token
        i_indices = (text_input_ids == self.tokenizer.convert_tokens_to_ids(" <i>")).nonzero()
        # replace the <i> token with the z_latents one by one
        inputs_embeds = self.text_encoder.embed_tokens(text_input_ids)
        if z_latents is None:
            pass
        elif i_indices.numel() > 0:
            # z_latents is (batch_size, num_images, num_tokens, num_channels)
            # inputs_embeds is (batch_size, num_tokens, num_channels)
            # we want to replace the <i> token with the z_latents one by one
            batch_size = inputs_embeds.shape[0]
            new_inputs_embeds = [None] * batch_size
            new_attention_masks = [None] * batch_size
            for batch_idx in range(batch_size):
                batch_embeds = inputs_embeds[batch_idx]
                batch_attention_mask = attention_mask[batch_idx]
                i_indices_batch = i_indices[i_indices[:, 0] == batch_idx]
                if i_indices_batch.shape[0] == 0:
                    # No <i> tokens found for this batch item
                    new_inputs_embeds[batch_idx] = batch_embeds.squeeze(0)
                    new_attention_masks[batch_idx] = batch_attention_mask.squeeze(0)
                else:
                    for i in range(i_indices_batch.shape[0]):
                        token_idx = i_indices_batch[i, 1]
                        prev_i_count = i
                        parts = []
                        if token_idx > 0:
                            parts.append(batch_embeds[:token_idx])
                        parts.append(z_latents[batch_idx, prev_i_count])
                        if token_idx + 1 < batch_embeds.shape[0]:
                            parts.append(batch_embeds[token_idx + 1 :])
                        batch_embeds = torch.cat(parts, dim=0)

                        mask_parts = []
                        if token_idx > 0:
                            mask_parts.append(batch_attention_mask[:token_idx])
                        mask_parts.append(
                            torch.ones(
                                z_latents.shape[2],
                                dtype=batch_attention_mask.dtype,
                                device=batch_attention_mask.device,
                            )
                        )
                        if token_idx + 1 < batch_attention_mask.shape[0]:
                            mask_parts.append(batch_attention_mask[token_idx + 1 :])
                        batch_attention_mask = torch.cat(mask_parts, dim=0)
                    new_inputs_embeds[batch_idx] = batch_embeds
                    new_attention_masks[batch_idx] = batch_attention_mask
            inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_inputs_embeds, batch_first=True)
            attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_masks, batch_first=True)
        else:
            # Original z_latents shape: [batch_size, num_images, num_tokens, dim]
            bsz, num_images, num_tokens, dim = z_latents.shape

            # Create BOI and EOI embeddings
            boi_embeds = self.text_encoder.embed_tokens(torch.tensor([self.boi_token_id], device=z_latents.device))
            eoi_embeds = self.text_encoder.embed_tokens(torch.tensor([self.eoi_token_id], device=z_latents.device))

            # Create repeated BOI and EOI tokens
            boi_tokens = boi_embeds.expand(bsz, num_images, 1, dim)
            eoi_tokens = eoi_embeds.expand(bsz, num_images, 1, dim)

            # Concatenate BOI, image tokens, and EOI along sequence dimension
            z_latents = torch.cat([boi_tokens, z_latents, eoi_tokens], dim=2)

            # Reshape to [batch_size, num_images * (num_tokens + 2), dim]
            z_latents = z_latents.reshape(bsz, -1, dim)
            inputs_embeds = torch.cat([z_latents, inputs_embeds], dim=1)

            # Update attention mask to match the new sequence length
            if attention_mask.shape[1] != inputs_embeds.shape[1]:
                attention_mask = torch.cat([torch.ones_like(z_latents[:, :, 0], dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)

        # gt_idx = torch.ones((inputs_embeds.shape[0], 1, inputs_embeds.shape[2]), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        # inputs_embeds = torch.cat((inputs_embeds, gt_idx), dim=1)
        # attention_mask = torch.cat((attention_mask, torch.ones_like(gt_idx[:, :, 0])), dim=1)

        latent_queries = self.latent_queries.repeat(inputs_embeds.shape[0], 1, 1)
        inputs_embeds = torch.cat([inputs_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0], dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

        prompt_embeds = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        return prompt_embeds[:, -self.config.num_pooled_tokens :], attention_mask[:, -self.config.num_pooled_tokens :]

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
