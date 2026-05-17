import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

# --- Rotary Positional Embedding Helpers ---

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Positional Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)  # [1, 1, S, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, S, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Core Transformer Components ---

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention module with dropout.
    """
    def __init__(self, embed_dim, num_heads, attn_dropout: float = 0.1, resid_dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)


        self.attn_dropout = float(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

    def forward(self, x, rotary_cos, rotary_sin):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        out = self.out_proj(out)
        return self.resid_drop(out)


class FeedForward(nn.Module):
    """
    Simple FeedForward network with SiLU (SwiGLU) activation and dropout.
    """
    def __init__(self, dim, hidden_dim, multiple_of=256, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.hidden_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w3(x)
        hidden = self.hidden_drop(hidden)
        out = self.w2(hidden)
        return self.out_drop(out)


class DecoderBlock(nn.Module):
    """
    A single Transformer block combining self-attention and a feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, attn_dropout: float = 0.1, resid_dropout: float = 0.1, ffn_dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
        self.ffn = FeedForward(dim=embed_dim, hidden_dim=None, dropout=ffn_dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, rotary_cos, rotary_sin):
        h = x + self.self_attn(self.norm1(x), rotary_cos, rotary_sin)
        out = h + self.ffn(self.norm2(h))
        return out


class AutoRegressiveDecoder(nn.Module):
    """
    The main auto-regressive decoder. It processes a unified sequence of ground,
    satellite, and action embeddings to predict the next zoom action.
    """
    def __init__(self, config, embed_dim, dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.grid_size = config.data.grid_size
        self.vocab_size = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        
        # An embedding layer for the discrete actions (patch indices)
        self.action_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                config.model.decoder_num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=dropout,
                ffn_dropout=dropout,
            )
            for _ in range(config.model.decoder_num_layers)
        ])
        
        self.final_norm = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, self.vocab_size, bias=False)

        # Precompute RoPE frequencies for positional encoding
        head_dim = embed_dim // config.model.decoder_num_heads
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, ground_global_feature, satellite_sequence_features, target_sequence):
        """
        Processes the sequence in a single pass for training using teacher-forcing.

        Args:
            ground_global_feature (torch.Tensor): Global embeddings of ground images. Shape: (B, D).
            satellite_sequence_features (torch.Tensor): Global embeddings of satellite images. Shape: (B, S, D).
            target_sequence (torch.Tensor): Ground-truth patch indices (actions). Shape: (B, S).
        """
        B, S, D = satellite_sequence_features.shape

        # Construct the full, interleaved input sequence for the transformer.
        # The sequence format is: [ground_feat, sat_feat_0, action_0, sat_feat_1, action_1, ...]
        ground_token = ground_global_feature.unsqueeze(1)
        action_embeddings = self.action_embed(target_sequence)
        
        # Interleave satellite features and action embeddings to form pairs.
        interleaved_sat_actions = torch.stack(
            [satellite_sequence_features, action_embeddings], dim=2
        ).view(B, 2 * S, D)
        
        # Prepend the ground token to create the final sequence of length (1 + 2*S).
        x = torch.cat([ground_token, interleaved_sat_actions], dim=1)

        x = self.embed_drop(x)

        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        t = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)
        cos, sin = torch.cos(t), torch.sin(t)
        rotary_cos = torch.cat((cos, cos), dim=-1)
        rotary_sin = torch.cat((sin, sin), dim=-1)

        h = x
        for block in self.blocks:
            h = block(h, rotary_cos, rotary_sin)
        
        # Select the hidden states corresponding to the satellite image positions.
        # Sequence: [g(0), s0(1), a0(2), s1(3), a1(4), s2(5), ...]
        prediction_features = h[:, 1::2, :]

        logits = self.output_head(self.final_norm(prediction_features))
        
        return logits