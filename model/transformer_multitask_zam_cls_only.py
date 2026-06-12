import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import numpy as np

from .factory import TransformerEncoderWrapper

class TransformerZAM_multitask_CLSOnly(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        num_classes: int = 22,
        dropout_rate_transformer: float = 0.2,
        dropout_rate_fc: float = 0.2,
        max_seq_len: int = 1000,
        positional_encoding_type: Literal["learnable", "sinusoidal", "relative"] = "sinusoidal",
    ):
        super().__init__()

        # Input projection and normalization
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Positional encoding module
        if positional_encoding_type == "learnable":
            from .factory import LearnablePositionalEncoding
            self.positional_encoding = LearnablePositionalEncoding(max_seq_len, hidden_size)
        elif positional_encoding_type == "sinusoidal":
            from .factory import SinusoidalPositionalEncoding
            self.positional_encoding = SinusoidalPositionalEncoding(max_seq_len, hidden_size)
        elif positional_encoding_type == "relative":
            from .factory import RelativePositionalEncoding
            self.positional_encoding = RelativePositionalEncoding(max_seq_len, hidden_size)
        else:
            raise ValueError("Invalid positional encoding type")

        self.encoding_type = positional_encoding_type

        # Transformer encoder
        self.encoder = TransformerEncoderWrapper(
            hidden_size, num_heads, dim_feedforward, num_layers, dropout_rate_transformer
        )

        # Sequence classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(128, num_classes),
        )
        
    def _prep_embeddings_and_mask(self, signal: torch.Tensor, length: torch.Tensor):
        """
        Returns:
          embedded: [B, T+1, D]  (with CLS prepended, PE added, pads zeroed)
          padding_mask: [B, T+1] (True = pad)
          seq_len: int (original T without CLS)
        """
        batch_size, seq_len, _ = signal.shape
        seq_len_plus_cls = seq_len + 1

        # Input projection
        embedded = self.input_projection(signal)
        embedded = self.input_layernorm(embedded)

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embedded = torch.cat((cls_token, embedded), dim=1)  # [B, T+1, D]

        # Build padding mask
        lengths_with_cls = length + 1
        padding_mask = torch.arange(seq_len_plus_cls, device=signal.device).expand(batch_size, -1) >= lengths_with_cls.unsqueeze(1)

        # Add positional encodings (excluding CLS)
        if self.encoding_type == "relative":
            rel_bias = self.positional_encoding(seq_len)       # [T, T, D]
            token_bias = rel_bias.sum(1)                       # [T, D]
            token_bias = token_bias.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, D]
            embedded[:, 1:] += token_bias * (~padding_mask[:, 1:]).unsqueeze(-1)
        else:
            abs_pe = self.positional_encoding(seq_len).to(embedded.device)  # [1, T, D]
            embedded[:, 1:] += abs_pe * (~padding_mask[:, 1:]).unsqueeze(-1)

        # Wipe padded slots completely
        embedded[padding_mask] = 0.0
        
        return embedded, padding_mask
    
    def forward(self, signal: torch.Tensor, length: torch.Tensor):
        """
        signal: Tensor of shape [batch_size, seq_len, input_dim]
        lengths: Tensor of shape [batch_size], unpadded lengths
        """
        embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)
    
        # Transformer encoder
        encoded = self.encoder(embedded, padding_mask)  # [B, T+1, D]

        # Outputs
        cls_representation = encoded[:, 0]  # [B, D]
        token_representation = encoded[:, 1:]  # [B, T, D]

        seq_logits = self.classifier(cls_representation)

        return {
            "classification": seq_logits,
        }

    @torch.no_grad()
    def get_cls_attention(self, signal: torch.Tensor, length: torch.Tensor, average_heads: bool = True):
        """
        Returns:
        cls_scores:        [B, H, T]   raw (scaled) dot-product scores from CLS -> tokens (no softmax)
        cls_attn:          [B, H, T]   softmaxed attention from CLS -> tokens (pads = 0)
        cls_attn_mean:     [B, T]      head-averaged softmaxed attention (if average_heads=True)
        """
        # Preserve training state — don't permanently switch to eval
        was_training = self.training
        self.eval()

        try:
            embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)  # [B, T+1, D]

            # Run all layers except the last
            x = embedded
            for layer in self.encoder.encoder.layers[:-1]:
                x = layer(x, src_key_padding_mask=padding_mask)

            last_layer = self.encoder.encoder.layers[-1]
            mha = last_layer.self_attn

            # Pre/post-norm handling
            if last_layer.norm_first:
                x_in = last_layer.norm1(x)
            else:
                x_in = x

            embed_dim = mha.embed_dim
            num_heads = mha.num_heads
            head_dim = embed_dim // num_heads
            scale = head_dim ** -0.5

            # Project to Q, K
            if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
                W_q, W_k, _ = mha.in_proj_weight.chunk(3, dim=0)
                b_q, b_k = (None, None) if mha.in_proj_bias is None else mha.in_proj_bias.chunk(3)[:2]
            else:
                W_q = mha.q_proj_weight
                W_k = mha.k_proj_weight
                b_q = None if mha.in_proj_bias is None else mha.in_proj_bias[:embed_dim]
                b_k = None if mha.in_proj_bias is None else mha.in_proj_bias[embed_dim:2 * embed_dim]

            Q = F.linear(x_in, W_q, b_q)  # [B, T+1, D]
            K = F.linear(x_in, W_k, b_k)  # [B, T+1, D]

            # Reshape to heads
            B, T_plus_1, _ = Q.shape
            Q = Q.view(B, T_plus_1, num_heads, head_dim).transpose(1, 2)  # [B, H, T+1, Dh]
            K = K.view(B, T_plus_1, num_heads, head_dim).transpose(1, 2)  # [B, H, T+1, Dh]

            # CLS query (pos 0) attending to tokens (pos 1..T)
            cls_q = Q[:, :, 0:1, :]   # [B, H, 1, Dh]
            keys  = K[:, :, 1:,  :]   # [B, H, T,   Dh]

            # Scaled dot-product scores
            cls_scores = torch.matmul(cls_q, keys.transpose(-2, -1)).squeeze(2) * scale  # [B, H, T]

            # Always apply padding mask for consistency (harmless when no padding)
            key_pad = padding_mask[:, 1:]           # [B, T], True = pad
            cls_scores = cls_scores.masked_fill(key_pad.unsqueeze(1), float('-inf'))

            # Softmax -> attention weights
            cls_attn = F.softmax(cls_scores, dim=-1)          # [B, H, T]
            cls_attn = torch.nan_to_num(cls_attn, nan=0.0)    # guard fully-padded rows

            cls_attn_mean = cls_attn.mean(dim=1) if average_heads else None  # [B, T]

            return cls_scores, cls_attn, cls_attn_mean

        finally:
            # Always restore training state, even if an exception occurs
            if was_training:
                self.train()
                
    def get_saliency(
        self,
        signal: torch.Tensor,                   # [B, T, input_size]
        length: torch.Tensor,                   # [B]
        target_class: torch.Tensor | None = None,  # [B] int, if None uses argmax
        absolute: bool = True,                  # take |grad| so sign doesn't matter
        normalize: bool = True,                 # min-max per sample over valid timesteps
    ) -> list[np.ndarray]:
        """
        Input-gradient saliency maps for a batch.

        Returns
        -------
        saliencies : list of B numpy arrays, each of shape (length[b],)
                     containing the saliency for the valid (unpadded) timesteps only.
        """
        was_training = self.training
        self.eval()

        try:
            signal = signal.clone()
            signal.requires_grad_(True)

            with torch.enable_grad():
                out    = self.forward(signal, length)         # {"classification": [B, C]}
                logits = out["classification"]                # [B, C]

                if target_class is None:
                    target_class = logits.argmax(dim=-1)      # [B]

                score = logits[torch.arange(len(target_class)), target_class].sum()
                score.backward()

            grads = signal.grad                               # [B, T, input_size]

            if absolute:
                grads = grads.abs()
            sal_full = grads.mean(dim=-1)                     # [B, T]

            saliencies = []
            for b in range(len(length)):
                L   = int(length[b].item())
                sal = sal_full[b, :L].detach().cpu().numpy()  # (L,)

                if normalize:
                    lo, hi = sal.min(), sal.max()
                    sal = (sal - lo) / (hi - lo + 1e-8)

                saliencies.append(sal)

            return saliencies

        finally:
            if was_training:
                self.train()