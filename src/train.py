"""CLI entrypoint for training baseline and joint TTT models.

Usage examples:

    # Quick sanity check with synthetic data (≈ 2 min)
    python -m src.train --synthetic --epochs 3 --batch_size 16

    # Full training on real data
    python -m src.train --parquet data/raw/btcusdt_1h.parquet \\
        --epochs 50 --aux_task mask --lambda_aux 1.0

    # Baseline (no aux task)
    python -m src.train --synthetic --epochs 10 --aux_task none
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CryptoRegimeDataset, prepare_dataset
from src.models import TTTModel
from src.ttt_learner import JointTrainer
from src.utils import seed_everything, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TTT crypto regime model")

    # data
    p.add_argument("--parquet", type=str, default=None,
                   help="Path to raw OHLCV parquet.")
    p.add_argument("--data_dir", type=str, default="data/processed",
                   help="Directory for processed dataset.pt.")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic data for pipeline testing.")
    p.add_argument("--synthetic_hours", type=int, default=4320,
                   help="Hours of synthetic data (default 4320 = 6 months).")

    # model
    p.add_argument("--aux_task", type=str, default="mask",
                   choices=["mask", "rotation", "none"],
                   help="Auxiliary self-supervised task.")
    p.add_argument("--mask_ratio", type=float, default=0.2,
                   help="Fraction of chart to mask (temporal masking).")
    p.add_argument("--num_groups", type=int, default=8,
                   help="GroupNorm groups.")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1,
                   help="Initial learning rate (SGD).")
    p.add_argument("--lambda_aux", type=float, default=1.0,
                   help="Weight on auxiliary loss.")
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    logger.info("Device: %s", device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Prepare dataset (if needed) ────────────────────────────────
    pt_path = Path(args.data_dir) / "dataset.pt"
    if not pt_path.exists():
        if args.synthetic:
            logger.info("Preparing synthetic dataset …")
            prepare_dataset(
                synthetic=True,
                synthetic_hours=args.synthetic_hours,
                output_dir=args.data_dir,
                seed=args.seed,
            )
        elif args.parquet:
            logger.info("Preparing dataset from %s …", args.parquet)
            prepare_dataset(
                parquet_path=args.parquet,
                output_dir=args.data_dir,
                seed=args.seed,
            )
        else:
            logger.error("No data found. Use --synthetic or --parquet.")
            sys.exit(1)

    # ── 2. Load dataset & create loaders ─────────────────────────────
    dataset = CryptoRegimeDataset(args.data_dir)
    train_ds, val_ds, test_ds = dataset.get_splits()

    logger.info("Dataset sizes — train: %d, val: %d, test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── 3. Create model ──────────────────────────────────────────────
    model = TTTModel(
        num_classes=2,
        aux_task=args.aux_task,
        num_groups=args.num_groups,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: TTTModel (aux=%s), %.2fM params", args.aux_task, n_params / 1e6)

    # ── 4. Create trainer ────────────────────────────────────────────
    trainer = JointTrainer(
        model=model,
        lr=args.lr,
        lambda_aux=args.lambda_aux if args.aux_task != "none" else 0.0,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=device,
    )

    # ── 5. Training loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_m = trainer.train_epoch(train_loader)

        # Validate (skip if val set is empty)
        if len(val_ds) > 0:
            val_m = trainer.validate(val_loader)
            val_loss = val_m.get("loss", float("inf"))
            val_acc = val_m.get("accuracy", 0.0)
        else:
            val_m = {}
            val_loss = float("inf")
            val_acc = train_m.get("accuracy", 0.0)  # fall back to train acc

        logger.info(
            "Epoch %3d/%d │ train_loss %.4f  train_acc %.3f │ "
            "val_loss %.4f  val_acc %.3f",
            epoch, args.epochs,
            train_m["loss"], train_m["accuracy"],
            val_loss, val_acc,
        )
        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        # checkpoint best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  → saved best checkpoint (val_acc=%.3f)", best_val_acc)

    # always save latest
    final_val_acc = val_acc if len(val_ds) > 0 else train_m.get("accuracy", 0.0)
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "val_accuracy": final_val_acc,
        "args": vars(args),
    }, ckpt_dir / "latest.pt")

    # save history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info("Training complete. Best val accuracy: %.3f", best_val_acc)


if __name__ == "__main__":
    main()
