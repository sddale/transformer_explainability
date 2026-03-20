"""
main.py — Full pipeline: train Tuned Lens, run both lenses, visualize.

Usage:
    python main.py                        # full pipeline
    python main.py --skip_training        # skip training (load saved probes)
    python main.py --prompt "your text"   # analyse custom prompt
"""

import argparse
import os
import json
from transformer_lens import HookedTransformer

from logit_lens import LogitLens
from tuned_lens import TunedLens
from train_tuned import train, TrainArgs
from util import get_device
from visualize import (
    plot_single_method,
    plot_comparison,
    plot_convergence,
    plot_entropy_comparison,
    build_html_dashboard,
)


DEFAULT_PROMPT = "The capital of France is Paris and the capital of Germany is"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser(
        description="Logit Lens vs Tuned Lens on pythia-14m"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text to analyse with both lenses",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training; load saved probes from checkpoints/",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device (default: auto-detect cuda → mps → cpu)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = get_device(args.device)

    # ─────────────────────────────────────────────────────────── #
    #  1. Load model
    # ─────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  Loading pythia-14m")
    print("=" * 60)
    model = HookedTransformer.from_pretrained("pythia-14m", device=str(device))
    model.eval()
    print(
        f"  Layers: {model.cfg.n_layers}  |  d_model: {model.cfg.d_model}  |  vocab: {model.cfg.d_vocab}"
    )

    # ─────────────────────────────────────────────────────────── #
    #  2. Logit Lens (no training needed)
    # ─────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  Running Logit Lens")
    print("=" * 60)
    logit_lens = LogitLens(model)
    logit_results = logit_lens.analyze(args.prompt)
    print(logit_lens.summary_table(logit_results))

    # ─────────────────────────────────────────────────────────── #
    #  3. Train (or load) Tuned Lens
    # ─────────────────────────────────────────────────────────── #
    probe_path = os.path.join(CHECKPOINT_DIR, "tuned_lens_probes.pt")
    tuned_lens = TunedLens(model).to(device)

    training_history = None
    if args.skip_training and os.path.exists(probe_path):
        print("\n[Main] Loading pre-trained probes …")
        tuned_lens.load(probe_path, map_location=device)
    else:
        print("\n" + "=" * 60)
        print("  Training Tuned Lens on Pile-10K")
        print("=" * 60)

        train_args = TrainArgs(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_samples=args.max_samples,
            output_dir=CHECKPOINT_DIR,
            device=args.device,
        )

        tuned_lens = train(train_args)
    hist_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            training_history = json.load(f)

    # ─────────────────────────────────────────────────────────── #
    #  4. Run Tuned Lens
    # ─────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  Running Tuned Lens")
    print("=" * 60)
    tuned_results = tuned_lens.analyze(args.prompt)

    # ─────────────────────────────────────────────────────────── #
    #  5. Generate all visualizations
    # ─────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  Generating visualizations")
    print("=" * 60)

    logit_b64 = plot_single_method(
        logit_results,
        "Logit Lens",
        save_path=os.path.join(OUTPUT_DIR, "logit_lens.png"),
    )

    tuned_b64 = plot_single_method(
        tuned_results,
        "Tuned Lens",
        save_path=os.path.join(OUTPUT_DIR, "tuned_lens.png"),
    )

    comparison_b64 = plot_comparison(
        logit_results,
        tuned_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison.png"),
    )

    convergence_b64 = plot_convergence(
        logit_results,
        tuned_results,
        save_path=os.path.join(OUTPUT_DIR, "convergence.png"),
    )

    entropy_b64 = plot_entropy_comparison(
        logit_results,
        tuned_results,
        save_path=os.path.join(OUTPUT_DIR, "entropy_comparison.png"),
    )

    # ─────────────────────────────────────────────────────────── #
    #  6. Build HTML dashboard
    # ─────────────────────────────────────────────────────────── #
    dashboard_path = os.path.join(OUTPUT_DIR, "dashboard.html")
    build_html_dashboard(
        logit_b64=logit_b64,
        tuned_b64=tuned_b64,
        comparison_b64=comparison_b64,
        convergence_b64=convergence_b64,
        entropy_b64=entropy_b64,
        prompt_text=args.prompt,
        training_history=training_history,
        save_path=dashboard_path,
    )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"\n  Outputs in: {os.path.abspath(OUTPUT_DIR)}/")
    print("    • logit_lens.png")
    print("    • tuned_lens.png")
    print("    • comparison.png")
    print("    • convergence.png")
    print("    • entropy_comparison.png")
    print("    • dashboard.html       ← open in browser for interactive view")
    print()


if __name__ == "__main__":
    main()
