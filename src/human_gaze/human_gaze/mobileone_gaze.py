"""MobileOne backbone with a two-head (yaw/pitch) gaze output - vendored and
trimmed to inference only, adapted from
github.com/yakhyo/gaze-estimation/models/mobileone.py (MIT-licensed,
Copyright (c) 2024 Yakhyokhuja Valikhujaev), which itself adapts Apple's
MobileOne (github.com/apple/ml-mobileone, Copyright (C) 2022 Apple Inc.).

Trimmed the way this project already trims every vendored inference-only
model (OSNet, 6DRepNet360): dropped the train-time multi-branch structure,
`reparameterize()`/`_fuse_bn_tensor()`, and the s1-s4 size variants - the
published Gaze360 checkpoint for `mobileone_s0` is already saved in fused,
inference-mode form (confirmed empirically: loads with zero missing/
unexpected keys directly into the inference-mode-only model below), so the
training-time branch-fusion machinery this project will never run is dead
weight here, not a safety net.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# (width_multipliers, num_conv_branches) - only the published Gaze360 variant.
_S0_WIDTH_MULTIPLIERS = (0.75, 1.0, 1.0, 2.0)


class _SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation (arxiv.org/abs/1709.01507) channel gate."""

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, int(in_channels * rd_ratio), kernel_size=1, bias=True)
        self.expand = nn.Conv2d(int(in_channels * rd_ratio), in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.size()
        gate = F.avg_pool2d(x, kernel_size=[h, w])
        gate = F.relu(self.reduce(gate))
        gate = torch.sigmoid(self.expand(gate))
        return x * gate.view(-1, c, 1, 1)


class _MobileOneBlock(nn.Module):
    """One already-fused (inference-mode) MobileOne conv block - no
    train-time multi-branch structure, since every weight this project
    loads is already fused (see module docstring)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.se = _SqueezeExcitationBlock(out_channels) if use_se else nn.Identity()
        self.activation = nn.ReLU()
        self.reparam_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.se(self.reparam_conv(x)))


class MobileOneGaze(nn.Module):
    """MobileOne-S0 backbone + two linear heads predicting yaw/pitch as a
    distribution over angle bins (see human_gaze/gaze_estimator.py for the
    bin -> degrees decode, matching the upstream repo's exact formula)."""

    def __init__(
        self,
        num_blocks_per_stage: List[int] = [2, 8, 10, 1],
        num_bins: int = 90,
        width_multipliers: Optional[List[float]] = None,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        width_multipliers = width_multipliers or list(_S0_WIDTH_MULTIPLIERS)
        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))

        self.stage0 = _MobileOneBlock(3, self.in_planes, kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], 0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], 0)
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]), num_blocks_per_stage[3], num_blocks_per_stage[3] if use_se else 0
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_yaw = nn.Linear(int(512 * width_multipliers[3]), num_bins)
        self.fc_pitch = nn.Linear(int(512 * width_multipliers[3]), num_bins)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int) -> nn.Sequential:
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for i, stride in enumerate(strides):
            use_se = i >= (num_blocks - num_se_blocks)
            blocks.append(
                _MobileOneBlock(
                    self.in_planes, self.in_planes, kernel_size=3, stride=stride, padding=1,
                    groups=self.in_planes, use_se=use_se,
                )
            )
            blocks.append(_MobileOneBlock(self.in_planes, planes, kernel_size=1, use_se=use_se))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.fc_yaw(x), self.fc_pitch(x)


def mobileone_s0_gaze(num_bins: int = 90) -> MobileOneGaze:
    return MobileOneGaze(width_multipliers=list(_S0_WIDTH_MULTIPLIERS), num_bins=num_bins)
