"""ResNet18-GN with Y-structure dual heads for Test-Time Training.

Architecture (following Sun et al., 2020):
    Shared Encoder : conv1 → GN → ReLU → MaxPool → Layer1 → Layer2 → Layer3
    Main Head      : Layer4 → AdaptiveAvgPool → FC(512 → 2)
    Aux Head (mask): Transposed-conv decoder  (256,14,14) → (3,224,224)
    Aux Head (rot) : AdaptiveAvgPool → FC(256 → 4)   [baseline]

Key design choices:
    - GroupNorm everywhere (critical for single-sample TTT; BN fails).
    - Split after 3rd residual group → deeper shared encoder.
    - Encoder features: (B, 256, 14, 14) for 224×224 input.

Re-implemented from scratch — no code from the original TTT repo.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────


class BasicBlock(nn.Module):
    """ResNet BasicBlock with GroupNorm instead of BatchNorm."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.GroupNorm(num_groups, planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


def _make_layer(
    in_planes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
    num_groups: int = 8,
) -> nn.Sequential:
    """Construct a residual layer with *num_blocks* BasicBlocks."""
    strides = [stride] + [1] * (num_blocks - 1)
    layers: list[nn.Module] = []
    cur = in_planes
    for s in strides:
        layers.append(BasicBlock(cur, planes, s, num_groups))
        cur = planes * BasicBlock.expansion
    return nn.Sequential(*layers)


# ── Shared encoder ───────────────────────────────────────────────────────


class SharedEncoder(nn.Module):
    """First three groups of ResNet18-GN.

    Input  : (B, 3, 224, 224)
    Output : (B, 256, 14, 14)
    """

    def __init__(self, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(64, 64, 2, stride=1, num_groups=num_groups)
        self.layer2 = _make_layer(64, 128, 2, stride=2, num_groups=num_groups)
        self.layer3 = _make_layer(128, 256, 2, stride=2, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ── Task-specific heads ──────────────────────────────────────────────────


class MainHead(nn.Module):
    """Main classification head: Layer4 → global pool → FC.

    Input  : (B, 256, 14, 14)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int = 2, num_groups: int = 8) -> None:
        super().__init__()
        self.layer4 = _make_layer(256, 512, 2, stride=2, num_groups=num_groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class MaskedPatchDecoder(nn.Module):
    """Auxiliary decoder for temporal masked-patch reconstruction.

    Upsamples encoder features back to image space so we can compute
    reconstruction loss on the masked (rightmost) region of the chart.

    Input  : (B, 256, 14, 14)
    Output : (B, 3, 224, 224)  — pixel values in [0, 1]
    """

    def __init__(self, num_groups: int = 8) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            # 14×14 → 28×28
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True),
            # 28×28 → 56×56
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
            # 56×56 → 112×112
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, 32), 32),
            nn.ReLU(inplace=True),
            # 112×112 → 224×224 (no Sigmoid: avoids vanishing grads in TTT)
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class RotationHead(nn.Module):
    """Auxiliary head for rotation prediction (baseline aux task).

    Input  : (B, 256, 14, 14)
    Output : (B, 4)  — logits for {0°, 90°, 180°, 270°}
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


# ── Full Y-structure model ───────────────────────────────────────────────


class TTTModel(nn.Module):
    """Y-structure model for Test-Time Training on crypto regime classification.

    Training:
        L_total = L_main(main_head(enc(x)), y) + λ · L_aux(aux_head(enc(x̃)), x)

    Test-Time Training (per sample):
        1. θ_enc ← θ_enc − α · ∇_{θ_enc} L_aux   (adapt encoder)
        2. ŷ = main_head(enc(x; θ_enc*))            (predict with adapted encoder)

    Args:
        num_classes: number of output classes (default 2: high / low vol).
        aux_task: ``'mask'`` | ``'rotation'`` | ``'none'``.
        num_groups: groups for GroupNorm (must divide all channel dims).
    """

    VALID_AUX = ("mask", "rotation", "none")

    def __init__(
        self,
        num_classes: int = 2,
        aux_task: str = "mask",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if aux_task not in self.VALID_AUX:
            raise ValueError(f"aux_task must be one of {self.VALID_AUX}, got '{aux_task}'")
        self.aux_task = aux_task

        self.encoder = SharedEncoder(num_groups=num_groups)
        self.main_head = MainHead(num_classes=num_classes, num_groups=num_groups)

        self.aux_head: Optional[nn.Module] = None
        if aux_task == "mask":
            self.aux_head = MaskedPatchDecoder(num_groups=num_groups)
        elif aux_task == "rotation":
            self.aux_head = RotationHead()

    # ── forward helpers ──────────────────────────────────────────────

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        """Full image → class logits."""
        return self.main_head(self.encoder(x))

    def forward_aux(self, x: torch.Tensor) -> torch.Tensor:
        """(Masked / rotated) image → reconstruction or rotation logits."""
        assert self.aux_head is not None, "No auxiliary head configured."
        return self.aux_head(self.encoder(x))

    def forward(
        self,
        x: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Joint forward pass for training.

        Args:
            x: clean images  (B, 3, H, W).
            x_aux: aux-task input (masked or rotated images).

        Returns:
            dict with ``'main_logits'`` and optionally ``'aux_output'``.
        """
        features = self.encoder(x)
        out: dict[str, torch.Tensor] = {"main_logits": self.main_head(features)}

        if x_aux is not None and self.aux_head is not None:
            features_aux = self.encoder(x_aux)
            out["aux_output"] = self.aux_head(features_aux)

        return out

    # ── parameter groups (for selective TTT updates) ─────────────────

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Return only shared-encoder parameters (updated during TTT)."""
        return list(self.encoder.parameters())

    def main_head_parameters(self) -> list[nn.Parameter]:
        return list(self.main_head.parameters())

    def aux_head_parameters(self) -> list[nn.Parameter]:
        return list(self.aux_head.parameters()) if self.aux_head else []


# ── Masking / rotation utilities ─────────────────────────────────────────


def create_temporal_mask(
    images: torch.Tensor,
    mask_ratio: float = 0.2,
    mask_mode: str = "random_slices",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask chart images for aux reconstruction.

    Args:
        images: (B, C, H, W) chart image batch.
        mask_ratio: fraction of image width to mask.
        mask_mode: "rightmost" = extrapolation (predict future);
                   "random_slices" = interpolation (mask middle, use context).

    Returns:
        masked_images: images with masked region zeroed.
        mask: (B, 1, H, W) binary mask (1 = masked pixel).
    """
    B, C, H, W = images.shape
    mask = torch.zeros(B, 1, H, W, device=images.device)

    if mask_mode == "rightmost":
        mask_cols = max(1, int(W * mask_ratio))
        mask[:, :, :, W - mask_cols :] = 1.0
    else:  # random_slices: interpolation, avoids extrapolation trap
        slice_width = max(1, int(W * mask_ratio / 3))
        pad = max(10, slice_width)
        lo, hi = pad, W - slice_width - pad
        if hi <= lo:
            # Fallback for small images: use rightmost
            mask[:, :, :, W - max(1, int(W * mask_ratio)) :] = 1.0
        else:
            for b in range(B):
                for _ in range(3):
                    start = torch.randint(lo, hi, (1,), device=images.device).item()
                    mask[b, :, :, start : start + slice_width] = 1.0

    masked_images = images.clone() * (1 - mask)
    return masked_images, mask


def apply_rotation(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random 0/90/180/270° rotation (baseline aux task).

    Returns:
        rotated: rotated image batch.
        labels: (B,) rotation class in {0, 1, 2, 3}.
    """
    B = images.shape[0]
    labels = torch.randint(0, 4, (B,), device=images.device)
    rotated = images.clone()
    for i in range(B):
        k = labels[i].item()
        if k > 0:
            rotated[i] = torch.rot90(images[i], k, dims=[1, 2])
    return rotated, labels
