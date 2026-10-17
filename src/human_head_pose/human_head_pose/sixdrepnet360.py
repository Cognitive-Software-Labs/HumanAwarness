# Adapted from github.com/thohemp/6DRepNet360 (sixdrepnet360/test.py,
# sixdrepnet360/utils.py) - only the architecture and the inference-time
# rotation-matrix -> Euler conversion are kept; training/dataset/plotting
# code is dropped since this project only ever runs inference.
#
# MIT License
#
# Copyright (c) 2023 Thorsten Hempel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math

import torch
import torch.nn as nn
import torchvision


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.clamp(v_mag, min=1e-8).view(-1, 1).expand_as(v)
    return v / v_mag


def _cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.stack((i, j, k), dim=1)


def compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """The paper's 6D rotation representation -> a proper 3x3 rotation matrix
    (Gram-Schmidt-style orthogonalization of the two predicted 3-vectors)."""
    x_raw, y_raw = poses[:, 0:3], poses[:, 3:6]
    x = _normalize_vector(x_raw)
    z = _normalize_vector(_cross_product(x, y_raw))
    y = _cross_product(z, x)
    return torch.stack((x, y, z), dim=2)


def compute_euler_angles_from_rotation_matrices(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> (x, y, z) Euler angles in radians, i.e. (pitch, yaw,
    roll) for this model's convention. Falls back to the gimbal-lock-safe
    formula when R[:, 0, 0]/R[:, 1, 0] alone can't pin down the rotation."""
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = (sy < 1e-6).float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = torch.zeros_like(z)

    return torch.stack(
        (x * (1 - singular) + xs * singular, y * (1 - singular) + ys * singular, z * (1 - singular) + zs * singular),
        dim=1,
    )


class SixDRepNet360(nn.Module):
    """ResNet-50-Bottleneck backbone + a linear head predicting the 6D
    rotation representation, converted to a rotation matrix in forward()."""

    def __init__(self):
        self.inplanes = 64
        super().__init__()
        block, layers = torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.linear_reg = nn.Linear(512 * block.expansion, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_reg(x)
        return compute_rotation_matrix_from_ortho6d(x)
