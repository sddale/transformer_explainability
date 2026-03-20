"""
Tuned Lens: Learn a lightweight affine probe per layer that translates
residual-stream activations into the space expected by the final
LayerNorm + Unembedding, yielding sharper intermediate predictions
than the vanilla logit lens.

Reference: "Eliciting Latent Predictions from Transformers with the Tuned Lens"
           (Belrose et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import os


class AffineProbe(nn.Module):
    """
    A single per-layer affine translator:  x ↦ Wx + b
    Initialized to the identity so the tuned lens starts out
    equivalent to the logit lens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


class TunedLens(nn.Module):
    """
    A collection of per-layer affine probes that are trained to minimise
    the KL divergence between their projected distributions and the
    model's final-layer output distribution.
    """

    def __init__(self, model: HookedTransformer):
        super().__init__()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.d_vocab = model.cfg.d_vocab

        # One affine probe per intermediate state (embed + each layer)
        self.probes = nn.ModuleList(
            [AffineProbe(self.d_model) for _ in range(self.n_layers + 1)]
        )

    # ------------------------------------------------------------------ #
    #  Forward helpers
    # ------------------------------------------------------------------ #
    def _get_residual_states(self, tokens: torch.Tensor, cache):
        """Extract residual-stream states from a TransformerLens cache."""
        states = []
        embed = cache["hook_embed"]
        if (
            "hook_pos_embed" in cache.cache_dict
            or "blocks.0.hook_pos_embed" in cache.cache_dict
        ):
            try:
                embed = embed + cache["hook_pos_embed"]
            except KeyError:
                pass  # RoPE model
        states.append(embed)
        for layer in range(self.n_layers):
            states.append(cache[f"blocks.{layer}.hook_resid_post"])
        return states

    def project(self, resid: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply probe → ln_final → unembed to get logits."""
        translated = self.probes[layer_idx](resid)
        normed = self.model.ln_final(translated)
        logits = normed @ self.model.W_U + self.model.b_U
        return logits

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #
    def training_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        KL(final_distribution || tuned_lens_distribution) averaged over
        all layers and positions.
        """
        with torch.no_grad():
            final_logits, cache = self.model.run_with_cache(tokens)
            assert final_logits is torch.tensor
            target_log_probs = F.log_softmax(final_logits, dim=-1)  # (B, S, V)
            states = self._get_residual_states(tokens, cache)

        total_loss = torch.tensor(0.0, device=tokens.device)
        for layer_idx, resid in enumerate(states):
            pred_logits = self.project(resid.detach(), layer_idx)  # (B, S, V)
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)
            # KL(target || pred)  = Σ target * (log target - log pred)
            kl = F.kl_div(
                pred_log_probs,
                target_log_probs.detach(),
                log_target=True,
                reduction="batchmean",
            )
            total_loss = total_loss + kl

        return total_loss / (self.n_layers + 1)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        """
        Mirrors the LogitLens.analyze() API so results can be compared
        directly.
        """
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens)
        token_strs = self.model.to_str_tokens(text)
        seq_len = tokens.shape[1]

        states = self._get_residual_states(tokens, cache)

        all_logits = []
        for layer_idx, resid in enumerate(states):
            logits = self.project(resid, layer_idx)
            all_logits.append(logits.squeeze(0))

        all_logits = torch.stack(all_logits, dim=0)
        probs = F.softmax(all_logits, dim=-1)
        top_probs, top_tokens = probs.max(dim=-1)

        top_strings = []
        for layer_idx in range(self.n_layers + 1):
            layer_strs = [
                self.model.to_string(top_tokens[layer_idx, pos].unsqueeze(0))
                for pos in range(seq_len)
            ]
            top_strings.append(layer_strs)

        final_pred = top_tokens[-1]
        target_probs = torch.stack(
            [
                probs[layer_idx, torch.arange(seq_len), final_pred]
                for layer_idx in range(self.n_layers + 1)
            ],
            dim=0,
        )

        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        return {
            "tokens": token_strs,
            "logits": all_logits.cpu(),
            "probs": probs.cpu(),
            "top_tokens": top_tokens.cpu(),
            "top_probs": top_probs.cpu(),
            "top_strings": top_strings,
            "target_probs": target_probs.cpu(),
            "entropy": entropy.cpu(),
        }

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(self.probes.state_dict(), path)
        print(f"[TunedLens] Probes saved to {path}")

    def load(self, path: str, map_location="cpu"):
        state = torch.load(path, map_location=str(map_location), weights_only=True)
        self.probes.load_state_dict(state)
        print(f"[TunedLens] Probes loaded from {path} → {map_location}")
