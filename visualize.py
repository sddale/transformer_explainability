"""
Visualization module for Logit Lens and Tuned Lens results.

Produces:
  1. Individual heatmaps for each method (top-token probability per layer x position)
  2. Entropy heatmaps for each method
  3. Side-by-side comparison panel
  4. A self-contained HTML dashboard with all plots embedded
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from typing import Optional


# ────────────────────────────────────────────────────────────────
#  Colour palettes
# ────────────────────────────────────────────────────────────────
_PROB_CMAP = "YlOrRd"
_ENTROPY_CMAP = "viridis"
_DIFF_CMAP = "RdBu_r"


def _fig_to_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _sanitize(s: str) -> str:
    """Clean token strings for display."""
    return s.replace("\n", "\\n").replace("\t", "\\t").strip()


# ────────────────────────────────────────────────────────────────
#  Individual method heatmap
# ────────────────────────────────────────────────────────────────
def plot_single_method(results: dict, title: str, save_path: Optional[str] = None):
    """
    Plot a two-row figure:
      row 1 - top-token probability heatmap with token labels
      row 2 - entropy heatmap
    """
    tokens = [_sanitize(t) for t in results["tokens"]]
    n_layers_plus_1 = results["top_probs"].shape[0]
    n_layers = n_layers_plus_1 - 1
    seq_len = len(tokens)

    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]
    top_probs_np = results["top_probs"].numpy()
    entropy_np = results["entropy"].numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(seq_len * 1.3, 10), n_layers_plus_1 * 0.7 + 3),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.patch.set_facecolor("#0e1117")

    # ── Top-token probability ──
    ax = axes[0]
    im = ax.imshow(top_probs_np, aspect="auto", cmap=_PROB_CMAP, vmin=0, vmax=1)
    ax.set_yticks(range(n_layers_plus_1))
    ax.set_yticklabels(layer_labels, fontsize=8, color="white")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, fontsize=7, rotation=45, ha="right", color="white")
    ax.set_title(f"{title} - Top-token Probability", fontsize=11, color="white", pad=10)
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")

    # Annotate cells with predicted token
    for layer in range(n_layers_plus_1):
        for p in range(seq_len):
            tok_str = _sanitize(results["top_strings"][layer][p])[:6]
            prob_val = top_probs_np[layer, p]
            colour = "black" if prob_val > 0.5 else "grey"
            ax.text(
                p,
                layer,
                tok_str,
                ha="center",
                va="center",
                fontsize=12,
                color=colour,
                fontweight="bold",
            )

    cb1 = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb1.ax.tick_params(colors="white", labelsize=7)

    # ── Entropy ──
    ax2 = axes[1]
    im2 = ax2.imshow(entropy_np, aspect="auto", cmap=_ENTROPY_CMAP)
    ax2.set_yticks(range(n_layers_plus_1))
    ax2.set_yticklabels(layer_labels, fontsize=8, color="white")
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(tokens, fontsize=7, rotation=45, ha="right", color="white")
    ax2.set_title(f"{title} - Entropy (nats)", fontsize=11, color="white", pad=10)
    ax2.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")

    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)
    cb2.ax.tick_params(colors="white", labelsize=7)

    fig.tight_layout(pad=1.5)

    b64 = _fig_to_b64(fig)
    if save_path:
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"[Viz] Saved → {save_path}")
    plt.close(fig)
    return b64


# ────────────────────────────────────────────────────────────────
#  Comparison figure
# ────────────────────────────────────────────────────────────────
def plot_comparison(
    logit_results: dict, tuned_results: dict, save_path: Optional[str] = None
):
    """
    Three-row figure:
      row 1 - Logit Lens top-prob
      row 2 - Tuned Lens top-prob
      row 3 - Δ probability (tuned − logit)
    """
    tokens = [_sanitize(t) for t in logit_results["tokens"]]
    n_layers_plus_1 = logit_results["top_probs"].shape[0]
    n_layers = n_layers_plus_1 - 1
    seq_len = len(tokens)
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    logit_probs = logit_results["top_probs"].numpy()
    tuned_probs = tuned_results["top_probs"].numpy()
    diff = tuned_probs - logit_probs

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(seq_len * 1.3, 10), n_layers_plus_1 * 1.0 + 4),
        gridspec_kw={"height_ratios": [2, 2, 2]},
    )
    fig.patch.set_facecolor("#0e1117")

    for idx, (data, cmap, vmin, vmax, label) in enumerate(
        [
            (logit_probs, _PROB_CMAP, 0, 1, "Logit Lens - Top-token Prob"),
            (tuned_probs, _PROB_CMAP, 0, 1, "Tuned Lens - Top-token Prob"),
            (diff, _DIFF_CMAP, -0.5, 0.5, "Δ Probability (Tuned − Logit)"),
        ]
    ):
        ax = axes[idx]
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(n_layers_plus_1))
        ax.set_yticklabels(layer_labels, fontsize=8, color="white")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, fontsize=7, rotation=45, ha="right", color="white")
        ax.set_title(label, fontsize=11, color="white", pad=8)
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")

        # Token annotations for the first two rows
        if idx < 2:
            src = logit_results if idx == 0 else tuned_results
            probs_arr = data
            for layer in range(n_layers_plus_1):
                for p in range(seq_len):
                    tok_str = _sanitize(src["top_strings"][layer][p])[:6]
                    prob_val = probs_arr[layer, p]
                    colour = "black" if prob_val > 0.5 else "gray"
                    ax.text(
                        p,
                        layer,
                        tok_str,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=colour,
                        fontweight="bold",
                    )

        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=7)

    fig.tight_layout(pad=1.5)
    b64 = _fig_to_b64(fig)
    if save_path:
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"[Viz] Saved → {save_path}")
    plt.close(fig)
    return b64


# ────────────────────────────────────────────────────────────────
#  Convergence / target-prob line chart
# ────────────────────────────────────────────────────────────────
def plot_convergence(
    logit_results: dict, tuned_results: dict, save_path: Optional[str] = None
):
    """
    Line chart: mean probability assigned to the *final-layer prediction*
    at each layer, for both methods.  Shows how quickly each lens
    converges to the model's actual output.
    """
    n_layers_plus_1 = logit_results["target_probs"].shape[0]
    layers = np.arange(n_layers_plus_1)
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers_plus_1 - 1)]

    logit_mean = logit_results["target_probs"].mean(dim=1).numpy()
    tuned_mean = tuned_results["target_probs"].mean(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(
        layers,
        logit_mean,
        "o-",
        color="#FF6B6B",
        linewidth=2,
        markersize=6,
        label="Logit Lens",
    )
    ax.plot(
        layers,
        tuned_mean,
        "s-",
        color="#4ECDC4",
        linewidth=2,
        markersize=6,
        label="Tuned Lens",
    )

    ax.fill_between(layers, logit_mean, tuned_mean, alpha=0.12, color="#4ECDC4")

    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=8, color="white")
    ax.set_xlabel("Layer", color="white", fontsize=10)
    ax.set_ylabel("Mean P(final prediction)", color="white", fontsize=10)
    ax.set_title("Convergence to Final Prediction", color="white", fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15, color="white")

    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    if save_path:
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"[Viz] Saved → {save_path}")
    plt.close(fig)
    return b64


# ────────────────────────────────────────────────────────────────
#  Entropy comparison line chart
# ────────────────────────────────────────────────────────────────
def plot_entropy_comparison(
    logit_results: dict, tuned_results: dict, save_path: Optional[str] = None
):
    """Mean entropy per layer for both methods."""
    n = logit_results["entropy"].shape[0]
    layers = np.arange(n)
    layer_labels = ["emb"] + [f"L{i}" for i in range(n - 1)]

    logit_ent = logit_results["entropy"].mean(dim=1).numpy()
    tuned_ent = tuned_results["entropy"].mean(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(
        layers,
        logit_ent,
        "o-",
        color="#FF6B6B",
        linewidth=2,
        markersize=6,
        label="Logit Lens",
    )
    ax.plot(
        layers,
        tuned_ent,
        "s-",
        color="#4ECDC4",
        linewidth=2,
        markersize=6,
        label="Tuned Lens",
    )

    ax.fill_between(layers, logit_ent, tuned_ent, alpha=0.12, color="#FF6B6B")

    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=8, color="white")
    ax.set_xlabel("Layer", color="white", fontsize=10)
    ax.set_ylabel("Mean Entropy (nats)", color="white", fontsize=10)
    ax.set_title("Distribution Entropy per Layer", color="white", fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    if save_path:
        fig.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"[Viz] Saved → {save_path}")
    plt.close(fig)
    return b64


# ────────────────────────────────────────────────────────────────
#  HTML dashboard
# ────────────────────────────────────────────────────────────────
def build_html_dashboard(
    logit_b64: str,
    tuned_b64: str,
    comparison_b64: str,
    convergence_b64: str,
    entropy_b64: str,
    prompt_text: str,
    training_history: Optional[dict] = None,
    save_path: str = "dashboard.html",
):
    """Assemble a self-contained HTML page with all visualizations."""

    # Optional training loss chart
    training_section = ""
    if training_history and training_history.get("batch_loss"):
        import matplotlib.pyplot as plt

        batch_losses = training_history["batch_loss"]
        epochs = training_history.get("epoch", [])
        n_epochs = len(epochs)
        bpe = training_history.get("batches_per_epoch")
        if not bpe:
            bpe = len(batch_losses) // max(n_epochs, 1)

        # Build fractional-epoch x values:
        #   batch 0   of epoch 1 → x = 1.0
        #   batch k   of epoch e → x = e + k / bpe
        #   batch 500 of epoch 2 (bpe=1000) → x = 2.5
        batch_x = []
        for epoch_idx in range(n_epochs):
            e = epochs[epoch_idx]  # 1-based epoch number
            start = epoch_idx * bpe
            end = min(start + bpe, len(batch_losses))
            for k in range(end - start):
                batch_x.append(e + k / bpe)
        # Handle any leftover batches beyond the expected count
        while len(batch_x) < len(batch_losses):
            batch_x.append(batch_x[-1] + 1 / max(bpe, 1))
        batch_x = np.array(batch_x[: len(batch_losses)])

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        # --- Raw per-batch loss (faint) ---
        ax.plot(
            batch_x,
            batch_losses,
            color="#E8A87C",
            linewidth=0.5,
            alpha=0.35,
            label="Batch loss",
            rasterized=True,
        )

        # --- Smoothed rolling average ---
        window = max(bpe // 4, 5)
        if len(batch_losses) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(batch_losses, kernel, mode="valid")
            half_w = window // 2
            smooth_x = batch_x[half_w : half_w + len(smoothed)]
            ax.plot(
                smooth_x,
                smoothed,
                color="#E8A87C",
                linewidth=1.8,
                alpha=0.85,
                label=f"Smoothed (w={window})",
            )

        # Epoch ticks at integer positions
        ax.set_xticks([e for e in epochs])
        ax.set_xticklabels([str(e) for e in epochs], color="white", fontsize=9)
        ax.set_xlim(epochs[0] - 0.05, epochs[-1] + 1 - 0.05)

        ax.set_xlabel("Epoch", color="white", fontsize=9)
        ax.set_ylabel("KL Loss", color="white", fontsize=9)
        ax.set_title("Tuned Lens Training Loss", color="white", fontsize=11)
        ax.legend(
            facecolor="#1a1a2e",
            edgecolor="#333",
            labelcolor="white",
            fontsize=8,
            loc="upper right",
        )
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.15, color="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        fig.tight_layout()
        train_b64 = _fig_to_b64(fig)
        plt.close(fig)
        training_section = f"""
        <section class="card">
            <h2>Training Progress</h2>
            <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
              Faint line = raw per-batch KL loss.  Solid orange = smoothed rolling average.
              X-axis ticks mark epoch boundaries; each batch is plotted at its fractional epoch position.
            </p>
            <img src="data:image/png;base64,{train_b64}" alt="Training loss">
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Logit Lens vs Tuned Lens — pythia-14m</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

  :root {{
    --bg: #0e1117;
    --card: #161b22;
    --accent1: #FF6B6B;
    --accent2: #4ECDC4;
    --accent3: #E8A87C;
    --text: #e6edf3;
    --muted: #8b949e;
    --border: #30363d;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}

  header {{
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }}

  header h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.4rem;
  }}

  header .subtitle {{
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
  }}

  .prompt-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-bottom: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--accent3);
  }}

  .prompt-box span {{ color: var(--muted); font-size: 0.8rem; }}

  nav {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }}

  nav button {{
    background: var(--card);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
  }}

  nav button:hover {{
    border-color: var(--accent2);
    color: var(--text);
  }}

  nav button.active {{
    background: var(--accent2);
    color: var(--bg);
    border-color: var(--accent2);
    font-weight: 600;
  }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}

  .card h2 {{
    font-size: 1.15rem;
    margin-bottom: 1rem;
    color: var(--accent2);
  }}

  .card img {{
    width: 100%;
    border-radius: 6px;
  }}

  .section {{ display: none; }}
  .section.active {{ display: block; }}

  .legend {{
    display: flex;
    gap: 2rem;
    justify-content: center;
    margin: 1rem 0;
    font-size: 0.85rem;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}

  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}

  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Logit Lens  vs  Tuned Lens</h1>
    <div class="subtitle">pythia-14m · TransformerLens · Pile-10K</div>
  </header>

  <div class="prompt-box">
    <span>Prompt ▸</span> {prompt_text}
  </div>

  <nav>
    <button class="active" onclick="show('logit')">Logit Lens</button>
    <button onclick="show('tuned')">Tuned Lens</button>
    <button onclick="show('compare')">Comparison</button>
    <button onclick="show('lines')">Convergence</button>
    <button onclick="show('training')">Training</button>
  </nav>

  <!-- Logit Lens -->
  <div id="logit" class="section active">
    <section class="card">
      <h2>Logit Lens Results</h2>
      <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Each cell shows the token the model would predict if we decoded the
        residual stream at that layer.  Colour = probability of that top token.
      </p>
      <img src="data:image/png;base64,{logit_b64}" alt="Logit Lens">
    </section>
  </div>

  <!-- Tuned Lens -->
  <div id="tuned" class="section">
    <section class="card">
      <h2>Tuned Lens Results</h2>
      <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Same idea, but each layer's residual stream is first passed through
        a learned affine probe before decoding — giving sharper predictions.
      </p>
      <img src="data:image/png;base64,{tuned_b64}" alt="Tuned Lens">
    </section>
  </div>

  <!-- Side-by-side comparison -->
  <div id="compare" class="section">
    <section class="card">
      <h2>Side-by-Side Comparison</h2>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent1)"></div> Logit Lens</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent2)"></div> Tuned Lens</div>
      </div>
      <img src="data:image/png;base64,{comparison_b64}" alt="Comparison">
    </section>
  </div>

  <!-- Convergence charts -->
  <div id="lines" class="section">
    <section class="card">
      <h2>Convergence to Final Prediction</h2>
      <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Average probability assigned to the model's final-layer prediction
        at each intermediate layer.  Higher = the lens recovers the final
        answer earlier.
      </p>
      <img src="data:image/png;base64,{convergence_b64}" alt="Convergence">
    </section>
    <section class="card">
      <h2>Entropy Comparison</h2>
      <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Lower entropy means the lens produces more confident (peaked)
        distributions.  The tuned lens typically has lower entropy at
        earlier layers.
      </p>
      <img src="data:image/png;base64,{entropy_b64}" alt="Entropy Comparison">
    </section>
  </div>

  <!-- Training -->
  <div id="training" class="section">
    {training_section if training_section else '<section class="card"><h2>Training</h2><p style="color:var(--muted)">No training history available.</p></section>'}
  </div>

  <footer>
    Logit Lens (nostalgebraist 2020) · Tuned Lens (Belrose et al. 2023) ·
    Model: EleutherAI/pythia-14m · Data: NeelNanda/pile-10k
  </footer>
</div>

<script>
function show(id) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    with open(save_path, "w") as f:
        f.write(html)
    print(f"[Viz] Dashboard saved → {save_path}")
    return save_path
