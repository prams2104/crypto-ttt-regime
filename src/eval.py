"""CLI entrypoint for evaluation and stress-testing.

Runs four evaluation modes on the test set and prints a comparison table:
    1. Baseline (no adaptation, fixed model)
    2. Joint training baseline (same model, no TTT)
    3. Standard TTT (adapt per sample, reset encoder)
    4. Online TTT (adapt sequentially, keep state)

Usage:
    python -m src.eval --checkpoint checkpoints/best.pt \\
        --ttt_steps 10 --ttt_lr 0.001

    # With entropy-adaptive LR
    python -m src.eval --checkpoint checkpoints/best.pt --entropy_adaptive
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CryptoRegimeDataset
from src.models import TTTModel
from src.ttt_learner import TTTAdaptor
from src.utils import (
    brier_score,
    expected_calibration_error,
    get_device,
    information_coefficient,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TTT crypto regime model")

    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                   help="Path to model checkpoint.")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (1 for standard TTT, larger for baseline).")

    # TTT settings
    p.add_argument("--ttt_steps", type=int, default=10,
                   help="Gradient steps per sample for standard TTT.")
    p.add_argument("--ttt_lr", type=float, default=0.01,
                   help="Base test-time learning rate. Paper uses 0.001 with SGD; 0.01-0.05 recommended for Adam.")
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--mask_mode", type=str, default=None,
                   help="Override checkpoint mask_mode (rightmost | random_slices).")
    p.add_argument("--ttt_optimizer", type=str, default="sgd", choices=["sgd", "adam"],
                   help="TTT optimizer (adam can escape flat MSE plateaus).")
    p.add_argument("--entropy_adaptive", action="store_true",
                   help="Enable entropy-adaptive TTT learning rate.")
    p.add_argument("--entropy_scale", type=float, default=2.0)
    p.add_argument("--entropy_gate_threshold", type=float, default=0.3,
                   help="Skip TTT when entropy < threshold * H_max (0.3 = 30%% of max).")

    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for P(high_vol). Lower improves recall.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    rv_values: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full evaluation metric suite."""
    preds = (probs[:, 1] > threshold).astype(int) if probs.ndim == 2 else (probs > threshold).astype(int)
    prob_pos = probs[:, 1] if probs.ndim == 2 else probs

    acc = (preds == labels).mean()

    # F1
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    ece = expected_calibration_error(prob_pos, labels)
    brier = brier_score(prob_pos, labels)

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "ece": float(ece),
        "brier": float(brier),
    }

    if rv_values is not None:
        ic = information_coefficient(prob_pos, rv_values)
        metrics["IC"] = ic

    return metrics


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print a comparison table of evaluation modes."""
    modes = list(results.keys())
    metric_names = list(results[modes[0]].keys())

    header = f"{'Mode':<20}" + "".join(f"{m:>10}" for m in metric_names)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for mode in modes:
        row = f"{mode:<20}"
        for m in metric_names:
            val = results[mode].get(m, float("nan"))
            row += f"{val:>10.4f}"
        print(row)
    print(sep)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    logger.info("Device: %s", device)

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    aux_task = train_args.get("aux_task", "mask")
    num_groups = train_args.get("num_groups", 8)

    model = TTTModel(num_classes=2, aux_task=aux_task, num_groups=num_groups).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d (val_acc=%.3f)",
                ckpt.get("epoch", -1), ckpt.get("val_accuracy", -1))

    # ── Load test data ───────────────────────────────────────────────
    dataset = CryptoRegimeDataset(args.data_dir)
    _, _, test_ds = dataset.get_splits()
    logger.info("Test set size: %d", len(test_ds))

    # ── 1. Baseline evaluation ───────────────────────────────────────
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    mask_mode = args.mask_mode or train_args.get("mask_mode", "random_slices")

    adaptor = TTTAdaptor(
        model=model,
        base_lr=args.ttt_lr,
        ttt_steps=args.ttt_steps,
        mask_ratio=args.mask_ratio,
        mask_mode=mask_mode,
        entropy_adaptive=args.entropy_adaptive,
        entropy_scale=args.entropy_scale,
        entropy_gate_threshold=args.entropy_gate_threshold,
        ttt_optimizer=args.ttt_optimizer,
        device=device,
    )

    logger.info("Running baseline evaluation …")
    baseline_out = adaptor.evaluate_baseline(test_loader)
    baseline_metrics = compute_metrics(
        baseline_out["probabilities"].numpy(),
        baseline_out["labels"].numpy(),
        baseline_out["rv_values"].numpy(),
        threshold=args.threshold,
    )

    results = {"Baseline": baseline_metrics}

    # ── 2. Standard TTT ──────────────────────────────────────────────
    if aux_task != "none":
        logger.info("Running standard TTT (steps=%d, lr=%.4f) …", args.ttt_steps, args.ttt_lr)
        all_preds, all_labels, all_probs, all_rv = [], [], [], []
        all_logs: list[list[dict]] = []

        # Standard TTT operates per sample (batch_size=1)
        single_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        for batch in tqdm(single_loader, desc="Standard TTT"):
            images, labels, rv = batch
            logits, logs = adaptor.adapt_and_predict(images)
            probs = F.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
            all_rv.append(rv)
            all_logs.append(logs)

        ttt_probs = torch.cat(all_probs).numpy()
        ttt_labels = torch.cat(all_labels).numpy()
        ttt_rv = torch.cat(all_rv).numpy()
        results["TTT (standard)"] = compute_metrics(
            ttt_probs, ttt_labels, ttt_rv, threshold=args.threshold,
        )

        # TTT debug: aux loss before vs after adaptation (first 100 samples)
        n_debug = min(100, len(all_logs))
        if n_debug > 0:
            init_loss = sum(all_logs[i][0]["aux_loss"] for i in range(n_debug)) / n_debug
            final_loss = sum(all_logs[i][-1]["aux_loss"] for i in range(n_debug)) / n_debug
            logger.info("TTT aux_loss (first %d samples): initial=%.4f → final=%.4f", n_debug, init_loss, final_loss)

        # ── 3. Online TTT ────────────────────────────────────────────
        logger.info("Running online TTT …")
        # Shuffle test set for online TTT (per paper §3); Fix 7: reproducible
        online_loader = DataLoader(
            test_ds, batch_size=1, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )

        # Need a fresh model copy for online TTT (modifies encoder permanently)
        model_online = TTTModel(num_classes=2, aux_task=aux_task, num_groups=num_groups).to(device)
        model_online.load_state_dict(ckpt["model_state_dict"])
        adaptor_online = TTTAdaptor(
            model=model_online,
            base_lr=args.ttt_lr,
            ttt_steps=1,
            mask_ratio=args.mask_ratio,
            mask_mode=mask_mode,
            entropy_adaptive=args.entropy_adaptive,
            entropy_scale=args.entropy_scale,
            entropy_gate_threshold=args.entropy_gate_threshold,
            ttt_optimizer=args.ttt_optimizer,
            device=device,
        )
        online_out = adaptor_online.evaluate_online(online_loader)
        results["TTT (online)"] = compute_metrics(
            online_out["probabilities"].numpy(),
            online_out["labels"].numpy(),
            online_out["rv_values"].numpy(),
            threshold=args.threshold,
        )

    # ── Print comparison ─────────────────────────────────────────────
    print_comparison_table(results)


if __name__ == "__main__":
    main()
