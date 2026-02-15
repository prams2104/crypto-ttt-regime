"""Training and Test-Time Adaptation loops.

Contains:
    - ``masked_reconstruction_loss``  — MSE over masked pixels only.
    - ``compute_entropy_lr``          — entropy-adaptive TTT learning rate.
    - ``JointTrainer``                — joint training on main + aux objectives.
    - ``TTTAdaptor``                  — test-time adaptation (standard & online).

Design follows Sun et al. (2020):
    Training : min_{θ_e, θ_m, θ_s}  Σ [ l_main(x,y; θ_m, θ_e) + λ · l_aux(x̃; θ_s, θ_e) ]
    TTT      : θ*_e = θ_e − α · ∇_{θ_e} l_aux(x̃; θ_s, θ_e)   (adapt encoder only)
               ŷ   = main_head(enc(x; θ*_e), θ_m)               (predict with adapted encoder)
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import TTTModel, create_temporal_mask, apply_rotation
from src.utils import MetricLogger, prediction_entropy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE reconstruction loss computed only on masked pixels.

    Args:
        reconstruction: (B, 3, H, W) decoder output.
        target: (B, 3, H, W) original (unmasked) images.
        mask: (B, 1, H, W) binary mask (1 = masked pixel).

    Returns:
        Scalar mean-squared error over masked region.
    """
    diff_sq = (reconstruction - target) ** 2        # (B, 3, H, W)
    masked_diff = diff_sq * mask                     # broadcast over C
    n_pixels = mask.sum() * target.shape[1]          # total masked elements
    return masked_diff.sum() / (n_pixels + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# Entropy-adaptive learning rate
# ═══════════════════════════════════════════════════════════════════════════


def compute_entropy_lr(
    base_lr: float,
    entropy: torch.Tensor,
    num_classes: int = 2,
    scale: float = 2.0,
    min_factor: float = 0.1,
) -> float:
    """Compute entropy-adaptive TTT learning rate.

    α(t) = base_lr × [ min_factor + (1 − min_factor) × scale × H(p) / H_max ]

    High entropy (confused model)  → larger adaptation step.
    Low entropy  (confident model) → smaller step (preserve stability).

    Args:
        base_lr: base test-time learning rate.
        entropy: scalar entropy of the current prediction.
        num_classes: for computing H_max = log(C).
        scale: multiplier on the normalised entropy.
        min_factor: minimum fraction of base_lr (floor).

    Returns:
        Adapted learning rate (float).
    """
    h_max = math.log(num_classes)
    h_norm = float(entropy.item()) / h_max if h_max > 0 else 0.0
    factor = min_factor + (1.0 - min_factor) * scale * min(h_norm, 1.0)
    return base_lr * factor


# ═══════════════════════════════════════════════════════════════════════════
# Joint Trainer
# ═══════════════════════════════════════════════════════════════════════════


class JointTrainer:
    """Joint training on main + self-supervised auxiliary objectives.

    Handles both ``mask`` (temporal masking) and ``rotation`` aux tasks.
    Learning-rate schedule: cosine annealing with warm restarts.

    Args:
        model: :class:`TTTModel` instance.
        lr: initial learning rate (SGD).
        lambda_aux: weight on auxiliary loss.
        mask_ratio: fraction of rightmost columns to mask.
        epochs: total epochs (for LR scheduler).
        device: torch device.
    """

    def __init__(
        self,
        model: TTTModel,
        lr: float = 0.1,
        lambda_aux: float = 1.0,
        mask_ratio: float = 0.2,
        epochs: int = 50,
        weight_decay: float = 5e-4,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.lambda_aux = lambda_aux
        self.mask_ratio = mask_ratio
        self.device = device

        self.criterion_main = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )

    # ── auxiliary loss dispatch ───────────────────────────────────────

    def _aux_loss(
        self,
        model: TTTModel,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary loss for the configured task."""
        if model.aux_task == "mask":
            masked_imgs, mask = create_temporal_mask(images, self.mask_ratio)
            recon = model.forward_aux(masked_imgs)
            return masked_reconstruction_loss(recon, images, mask)
        elif model.aux_task == "rotation":
            rotated, rot_labels = apply_rotation(images)
            rot_logits = model.forward_aux(rotated)
            return F.cross_entropy(rot_logits, rot_labels)
        return torch.tensor(0.0, device=images.device)

    # ── training epoch ────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch over *loader*.

        Each batch yields ``(images, labels, rv_values)`` from
        :class:`CryptoRegimeDataset`.

        Returns:
            dict of average metrics (loss, loss_main, loss_aux, accuracy).
        """
        self.model.train()
        meter = MetricLogger()

        for batch in loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            # ── forward (main) ──
            logits = self.model.forward_main(images)
            loss_main = self.criterion_main(logits, labels)

            # ── forward (aux) ──
            loss_aux = self._aux_loss(self.model, images)
            loss = loss_main + self.lambda_aux * loss_aux

            # ── backward ──
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ── metrics ──
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            meter.update(
                loss=loss.item(),
                loss_main=loss_main.item(),
                loss_aux=loss_aux.item(),
                accuracy=acc,
            )

        self.scheduler.step()
        return meter.summary()

    # ── validation ────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on *loader* (no TTT — fixed model)."""
        self.model.eval()
        meter = MetricLogger()

        for batch in loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model.forward_main(images)
            loss = self.criterion_main(logits, labels)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            meter.update(loss=loss.item(), accuracy=acc)

        return meter.summary()


# ═══════════════════════════════════════════════════════════════════════════
# TTT Adaptor (test-time training)
# ═══════════════════════════════════════════════════════════════════════════


class TTTAdaptor:
    """Test-time adaptation via auxiliary self-supervised loss.

    Supports:
        - **Standard TTT**: adapt on each sample independently, reset encoder.
        - **Online TTT**: adapt sequentially, keep encoder state.
        - **Entropy-adaptive LR**: scale α(t) by prediction entropy.

    Args:
        model: jointly-trained :class:`TTTModel`.
        base_lr: base test-time learning rate (paper default 0.001).
        ttt_steps: gradient steps per sample in standard mode (paper: 10).
        mask_ratio: fraction of image to mask for aux task.
        entropy_adaptive: whether to scale LR by entropy.
        entropy_scale: multiplier inside ``compute_entropy_lr``.
        device: torch device.
    """

    def __init__(
        self,
        model: TTTModel,
        base_lr: float = 0.001,
        ttt_steps: int = 10,
        mask_ratio: float = 0.2,
        entropy_adaptive: bool = True,
        entropy_scale: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.base_lr = base_lr
        self.ttt_steps = ttt_steps
        self.mask_ratio = mask_ratio
        self.entropy_adaptive = entropy_adaptive
        self.entropy_scale = entropy_scale
        self.device = device

        # TTT optimizer: SGD, zero momentum / weight-decay (per paper §3)
        self._ttt_opt = torch.optim.SGD(
            model.encoder.parameters(), lr=base_lr, momentum=0.0, weight_decay=0.0,
        )

    # ── internal helpers ─────────────────────────────────────────────

    def _save_encoder(self) -> dict[str, torch.Tensor]:
        return {n: p.data.clone() for n, p in self.model.encoder.named_parameters()}

    def _restore_encoder(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.model.encoder.named_parameters():
                param.data.copy_(state[name])

    def _current_lr(self, images: torch.Tensor) -> float:
        """Optionally compute entropy-adaptive LR."""
        if not self.entropy_adaptive:
            return self.base_lr
        with torch.no_grad():
            logits = self.model.forward_main(images)
            probs = F.softmax(logits, dim=-1)
            ent = prediction_entropy(probs).mean()
        return compute_entropy_lr(
            self.base_lr, ent,
            num_classes=logits.shape[-1],
            scale=self.entropy_scale,
        )

    def _ttt_step(self, images: torch.Tensor) -> float:
        """Single TTT gradient step; returns aux loss value."""
        if self.model.aux_task == "mask":
            masked, mask = create_temporal_mask(images, self.mask_ratio)
            recon = self.model.forward_aux(masked)
            loss = masked_reconstruction_loss(recon, images, mask)
        elif self.model.aux_task == "rotation":
            rotated, rot_labels = apply_rotation(images)
            rot_logits = self.model.forward_aux(rotated)
            loss = F.cross_entropy(rot_logits, rot_labels)
        else:
            raise RuntimeError("TTT requires an auxiliary task.")

        self._ttt_opt.zero_grad()
        loss.backward()
        self._ttt_opt.step()
        return loss.item()

    # ── standard TTT ─────────────────────────────────────────────────

    def adapt_and_predict(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Standard TTT: adapt encoder → predict → reset encoder.

        Args:
            images: (B, 3, H, W) batch (typically B=1).

        Returns:
            logits: (B, C) main-task logits with adapted encoder.
            logs: per-step adaptation diagnostics.
        """
        images = images.to(self.device)
        saved = self._save_encoder()
        logs: list[dict] = []

        self.model.train()
        for step in range(self.ttt_steps):
            lr = self._current_lr(images)
            for pg in self._ttt_opt.param_groups:
                pg["lr"] = lr
            aux_loss = self._ttt_step(images)
            logs.append({"step": step, "aux_loss": aux_loss, "lr": lr})

        # predict with adapted encoder
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward_main(images)

        # restore original encoder
        self._restore_encoder(saved)
        return logits, logs

    # ── online TTT ───────────────────────────────────────────────────

    def evaluate_online(
        self,
        loader: DataLoader,
    ) -> dict[str, torch.Tensor | list]:
        """Online TTT: keep encoder state across sequential samples.

        One gradient step per sample.  Encoder is NOT reset between samples.
        To avoid ordering artefacts, caller should shuffle the test set
        (per the paper §3).

        Returns:
            dict with 'predictions', 'labels', 'probabilities', 'logs'.
        """
        all_preds, all_labels, all_probs = [], [], []
        logs: list[dict] = []

        self.model.train()
        for batch in loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # entropy-adaptive LR
            lr = self._current_lr(images)
            for pg in self._ttt_opt.param_groups:
                pg["lr"] = lr

            # one TTT step
            aux_loss = self._ttt_step(images)

            # predict with updated model
            self.model.eval()
            with torch.no_grad():
                logits = self.model.forward_main(images)
                probs = F.softmax(logits, dim=-1)
            self.model.train()

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            logs.append({"aux_loss": aux_loss, "lr": lr})

        return {
            "predictions": torch.cat(all_preds),
            "labels": torch.cat(all_labels),
            "probabilities": torch.cat(all_probs),
            "logs": logs,
        }

    # ── baseline evaluation (no TTT) ─────────────────────────────────

    @torch.no_grad()
    def evaluate_baseline(
        self,
        loader: DataLoader,
    ) -> dict[str, torch.Tensor]:
        """Standard evaluation — no test-time adaptation."""
        self.model.eval()
        all_preds, all_labels, all_probs, all_rv = [], [], [], []

        for batch in loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            rv     = batch[2].to(self.device)

            logits = self.model.forward_main(images)
            probs  = F.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_rv.append(rv.cpu())

        return {
            "predictions":   torch.cat(all_preds),
            "labels":        torch.cat(all_labels),
            "probabilities": torch.cat(all_probs),
            "rv_values":     torch.cat(all_rv),
        }
