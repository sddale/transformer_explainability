"""
Logit Lens: Inspect intermediate representations of a transformer model
by projecting residual stream activations through the final LayerNorm + Unembedding.

Reference: "interpreting GPT: the logit lens" (nostalgebraist, 2020)
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


class LogitLens:
    """
    Applies the model's own unembedding head (ln_final + W_U) to every
    intermediate residual-stream state, producing a layer x position
    distribution over the vocabulary at each point in the forward pass.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.model.eval()
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.d_vocab = model.cfg.d_vocab

    # ------------------------------------------------------------------ #
    #  Core analysis
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        """
        Run the logit lens on *text* and return a results dict:
            tokens       : list[str]           - tokenized input
            logits       : (n_layers+1, seq, vocab) - raw logits per layer
            probs        : (n_layers+1, seq, vocab) - softmax probs
            top_tokens   : (n_layers+1, seq)   - argmax token ids
            top_probs    : (n_layers+1, seq)    - probability of top token
            top_strings  : list[list[str]]      - decoded top tokens
            target_probs : (n_layers+1, seq)    - prob assigned to final-layer prediction
            entropy      : (n_layers+1, seq)    - Shannon entropy of distribution
        Layer index 0 = after embedding, index L = after final layer.
        """
        tokens = self.model.to_tokens(text)  # (1, seq)
        _, cache = self.model.run_with_cache(tokens)

        token_strs = self.model.to_str_tokens(text)
        seq_len = tokens.shape[1]

        # Collect residual stream states: after embed + after each layer
        # Pythia (and other RoPE models) have no learned pos embed —
        # rotary PE is applied inside attention, so we just use hook_embed.
        resid_states = []
        embed = cache["hook_embed"]
        if (
            "hook_pos_embed" in cache.cache_dict
            or "blocks.0.hook_pos_embed" in cache.cache_dict
        ):
            try:
                embed = embed + cache["hook_pos_embed"]
            except KeyError:
                pass  # RoPE model — positional info is in attention
        resid_states.append(embed)  # layer 0
        for layer in range(self.n_layers):
            resid_states.append(cache[f"blocks.{layer}.hook_resid_post"])

        all_logits = []
        for resid in resid_states:
            normed = self.model.ln_final(resid)  # (1, seq, d_model)
            logits = normed @ self.model.W_U + self.model.b_U  # (1, seq, vocab)
            all_logits.append(logits.squeeze(0))

        all_logits = torch.stack(all_logits, dim=0)  # (n_layers+1, seq, vocab)
        probs = F.softmax(all_logits, dim=-1)

        top_probs, top_tokens = probs.max(dim=-1)  # (n_layers+1, seq)

        # Decode top tokens to strings
        top_strings = []
        for layer_idx in range(self.n_layers + 1):
            layer_strs = [
                self.model.to_string(top_tokens[layer_idx, pos].unsqueeze(0))
                for pos in range(seq_len)
            ]
            top_strings.append(layer_strs)

        # Final-layer prediction ids
        final_pred = top_tokens[-1]  # (seq,)
        target_probs = torch.stack(
            [
                probs[layer_idx, torch.arange(seq_len), final_pred]
                for layer_idx in range(self.n_layers + 1)
            ],
            dim=0,
        )

        # Entropy in nats
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
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    def get_top_k_at_layer(self, results: dict, layer: int, position: int, k: int = 10):
        """Return the top-k tokens and probabilities at a specific (layer, position)."""
        probs = results["probs"][layer, position]
        top_probs, top_ids = probs.topk(k)
        top_strs = [self.model.to_string(tid.unsqueeze(0)) for tid in top_ids]
        return list(zip(top_strs, top_probs.tolist()))

    def summary_table(self, results: dict) -> str:
        """Pretty-print the top prediction at every (layer, position)."""
        tokens = results["tokens"]
        lines = [f"{'Layer':<8}" + "".join(f"{t:<14}" for t in tokens)]
        lines.append("-" * len(lines[0]))
        for layer_idx in range(self.n_layers + 1):
            label = "embed" if layer_idx == 0 else f"L{layer_idx - 1}"
            row = f"{label:<8}"
            for pos in range(len(tokens)):
                tok = results["top_strings"][layer_idx][pos]
                prob = results["top_probs"][layer_idx, pos].item()
                row += f"{tok}({prob:.2f})".ljust(14)
            lines.append(row)
        return "\n".join(lines)
