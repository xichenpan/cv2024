import logging
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

FLASH_ATTN_AVAILABLE = True
try:
    from flash_attn import flash_attn_func
except ImportError:
    FLASH_ATTN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PFRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.

        """
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#############################################################################
#             Embedding Layers for Timesteps and condition latents               #
#############################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        # Initialize the weights with normal distribution
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class LatentEmbedder(nn.Module):
    r"""Embeds condtion latents into vector representations. Also handles
    dropout for classifier-free guidance.
    """

    def __init__(self, latent_embedding_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                latent_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        # Initialize the weights with normal distribution
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
        self.latent_embedding_size = latent_embedding_size

    def forward(self, latents):
        train = not ((latents[0] == 0).all() or (latents[-1] == 0).all())
        embeddings = self.mlp(latents)
        return embeddings


#############################################################################
#                               Core DiT Model                              #
#############################################################################


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], qk_norm: bool):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads (int): Number of local query heads.
            n_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (Linear): Linear transformation for queries.
            wk (Linear): Linear transformation for keys.
            wv (Linear): Linear transformation for values.
            wo (Linear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        nn.init.xavier_uniform_(self.wq.weight)

        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        nn.init.xavier_uniform_(self.wk.weight)

        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        nn.init.xavier_uniform_(self.wv.weight)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.wo.weight)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        if dtype in [torch.float16, torch.bfloat16] and FLASH_ATTN_AVAILABLE:
            output = flash_attn_func(xq, xk, xv, dropout_p=0.0, causal=False)
        else:
            n_rep = self.n_heads // self.n_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                xk.permute(0, 2, 1, 3),
                xv.permute(0, 2, 1, 3),
                dropout_p=0.0,
                is_causal=False,
            ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first
                layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.w2.weight)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w3.weight)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlockSandwichNorm(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value in the FeedForward block.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension in the FeedForward block. Defaults to None.
            norm_eps (float): A small value added to the norm layer
                denominators to avoid division-by-zero.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm1 = PFRMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = PFRMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(dim, 1024),
                4 * dim,
                bias=True,
            ),
        )

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            h = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm(
                self.attention(
                    self.attention_norm1(x) * (1 + scale_msa.unsqueeze(1)),
                    freqs_cis,
                )
            )
            out = h + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm(
                self.feed_forward(
                    self.ffn_norm1(h) * (1 + scale_mlp.unsqueeze(1)),
                )
            )

        else:
            raise NotImplementedError

        return out


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                2 * hidden_size,
                bias=True,
            ),
        )

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class NextDiTConfig(PretrainedConfig):
    model_type = "dit-llama"

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 1536,
        n_layers: int = 16,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        latent_embedding_size: int = 1152,
        learn_sigma: bool = True,
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
        self.latent_embedding_size = latent_embedding_size
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm
        self._gradient_checkpointing = _gradient_checkpointing


class NextDiT(PreTrainedModel):
    config_class = NextDiTConfig

    def __init__(
        self,
        config: NextDiTConfig,
    ) -> None:
        super().__init__(config)
        self.learn_sigma = config.learn_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.input_size = config.input_size
        self.patch_size = config.patch_size
        self._gradient_checkpointing = config._gradient_checkpointing

        self.x_embedder = nn.Linear(
            config.patch_size * config.patch_size * config.in_channels,
            config.dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.t_embedder = TimestepEmbedder(min(config.dim, 1024))
        self.z_embedder = LatentEmbedder(config.latent_embedding_size, min(config.dim, 1024))

        self.layers = nn.ModuleList(
            [
                TransformerBlockSandwichNorm(
                    layer_id,
                    config.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.multiple_of,
                    config.ffn_dim_multiplier,
                    config.norm_eps,
                    config.qk_norm,
                )
                for layer_id in range(config.n_layers)
            ]
        )
        self.final_layer = FinalLayer(config.dim, config.patch_size, self.out_channels)

        assert (config.dim // config.n_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"
        self.freqs_cis = NextDiT.precompute_freqs_cis(
            config.dim // config.n_heads,
            384,
            # theta=100.
        )

        self.dim = config.dim
        self.n_heads = config.n_heads

    def unpatchify(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        B = x.shape[0]
        C = self.out_channels
        P = self.patch_size

        x = x.reshape(shape=(B, H, W, P, P, C))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(B, C, H * P, W * P))
        return imgs

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        # assert (H, W) == (self.input_size, self.input_size)
        pH = pW = self.patch_size
        x = x.view(B, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)

        return x, H // pH, W // pW, self.freqs_cis[: H // pH, : W // pW].flatten(0, 1).unsqueeze(0)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: float = 1.0, ntk_factor: float = 1.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        logger.info(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 1).repeat(1, end, 1, 1)
        freqs_cis_w = freqs_cis.view(1, end, dim // 4, 1).repeat(end, 1, 1, 1)
        freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1).flatten(2)
        return freqs_cis

    def forward(self, x, timestep, z_latents, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent
           representations of images)
        t: (N,) tensor of diffusion timesteps
        z: (N,) tensor of condition latents
        """

        x, H, W, freqs_cis = self.patchify(x)
        freqs_cis = freqs_cis.to(x.device)
        x = self.x_embedder(x)

        t = self.t_embedder(timestep)  # (N, D)
        z = self.z_embedder(z_latents)  # (N, D)
        adaln_input = t + z

        for layer in self.layers:
            if self.training and self._gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, freqs_cis, adaln_input, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, H, W)  # (N, out_channels, H, W)
        return x
