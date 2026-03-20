"""
Train the Tuned Lens affine probes on the Pile-10K dataset.

Usage:
    python train.py [--epochs 3] [--lr 1e-3] [--batch_size 4] [--max_length 128]
"""

import argparse
import torch
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tuned_lens import TunedLens
from util import get_device
from tqdm import tqdm
from typing import Optional
import os
import json
import time


@dataclass
class TrainArgs:
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 4
    max_length: int = 128
    max_samples: int = 5000
    output_dir: str = "checkpoints"
    device: str | None = None


class PileTokenDataset(Dataset[torch.Tensor]):
    """Tokenise Pile-10K texts on the fly and return fixed-length windows.

    Tokens are always stored on CPU to conserve accelerator memory;
    the training loop moves each batch to the target device.
    """

    def __init__(
        self, model: HookedTransformer, max_length: int = 128, max_samples: int = 5000
    ):
        print("[Data] Loading NeelNanda/pile-10k …")
        raw = load_dataset("NeelNanda/pile-10k", split="train")
        self.tokens_list = []
        print(
            f"[Data] Tokenizing {min(len(raw), max_samples)} documents (max_length={max_length}) …"
        )
        for i, example in enumerate(raw):
            if i >= max_samples:
                break
            text = example["text"]
            if len(text.strip()) < 20:
                continue
            toks = model.to_tokens(text, prepend_bos=True).squeeze(0).cpu()
            # Slice to max_length; skip very short sequences
            if toks.shape[0] >= 16:
                self.tokens_list.append(toks[:max_length])
        print(f"[Data] Prepared {len(self.tokens_list)} token sequences.")

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        return self.tokens_list[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    max_len = max(t.shape[0] for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(batch):
        padded[i, : t.shape[0]] = t
    return padded


def train(args: TrainArgs) -> TunedLens:
    device = get_device(args.device)

    # -------------------------------------------------------------- #
    #  Load model
    # -------------------------------------------------------------- #
    print("[Train] Loading pythia-14m via TransformerLens …")
    # TransformerLens expects a device string, not torch.device
    model = HookedTransformer.from_pretrained("pythia-14m", device=str(device))
    model.eval()

    # -------------------------------------------------------------- #
    #  Build tuned lens + dataset
    # -------------------------------------------------------------- #
    tuned_lens = TunedLens(model).to(device)
    dataset = PileTokenDataset(
        model, max_length=args.max_length, max_samples=args.max_samples
    )

    # pin_memory speeds up CPU→CUDA transfers; not relevant for MPS
    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin,
        num_workers=0,  # safest default across platforms
    )

    optimizer = optim.AdamW(
        tuned_lens.probes.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # Enable mixed-precision on CUDA for speed (MPS float16 autocast
    # is supported on PyTorch ≥2.1 but can be flaky — keep float32).
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # -------------------------------------------------------------- #
    #  Training loop
    # -------------------------------------------------------------- #
    history: dict[str, list | Optional[int]] = {
        "epoch": [],
        "loss": [],
        "lr": [],
        "batch_loss": [],
        "batches_per_epoch": None,
    }
    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Training Tuned Lens probes")
    print(f"  Device : {device}")
    print(f"  AMP    : {'on' if use_amp else 'off'}")
    print(f"  Layers : {tuned_lens.n_layers + 1} probes")
    print(f"  Params : {sum(p.numel() for p in tuned_lens.probes.parameters()):,}")
    print(f"  Data   : {len(dataset)} sequences")
    print(f"  Epochs : {args.epochs}   Batch: {args.batch_size}   LR: {args.lr}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        tuned_lens.probes.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch_tokens in pbar:
            batch_tokens = batch_tokens.to(device, non_blocking=pin)

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = tuned_lens.training_loss(batch_tokens)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(tuned_lens.probes.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = tuned_lens.training_loss(batch_tokens)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tuned_lens.probes.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            batch_loss_val = loss.item()
            history["batch_loss"].append(batch_loss_val)
            epoch_loss += batch_loss_val
            n_batches += 1
            pbar.set_postfix(
                loss=f"{batch_loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  → Epoch {epoch}  avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "tuned_lens_probes.pt")
            tuned_lens.save(save_path)

    # Save training history
    history["batches_per_epoch"] = len(loader)
    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Training history saved to {hist_path}")
    print(f"[Train] Best loss: {best_loss:.4f}")

    return tuned_lens


def main():
    parser = argparse.ArgumentParser(description="Train Tuned Lens probes")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device (default: auto-detect cuda → mps → cpu)",
    )
    ns = parser.parse_args()
    train(TrainArgs(**vars(ns)))


if __name__ == "__main__":
    main()
