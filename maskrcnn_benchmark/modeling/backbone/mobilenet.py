import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import Counter
from maskrcnn_benchmark.layers.misc import Conv2d
from maskrcnn_benchmark.layers import FrozenBatchNorm2d


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size, bias=False),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=(height, width)).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nonLinear, SE, exp_size, dropout_rate=1.0):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        self.dropout_rate = dropout_rate
        padding = (kernel_size - 1) // 2
        self.use_connect = (stride == 1 and in_channels == out_channels)

        # RE: ReLU;HS: h_swish
        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            FrozenBatchNorm2d(exp_size),
            activation(inplace=True)
        )

        self.depth_conv = nn.Sequential(
            Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=padding, groups=exp_size),
            FrozenBatchNorm2d(exp_size)
        )

        # Squeeze-and-Excite
        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            FrozenBatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNet V2
        out = self.conv(x)
        out = self.depth_conv(out)

        if self.SE:
            out = self.squeeze_block(out)

        out = self.point_conv(out)

        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, cfg, multiplier=1.0):
        super(MobileNetV3, self).__init__()
        self.activation_HS = nn.ReLU6(inplace=True)

        bneck2_in_channels = cfg.MODEL.MOBILENETS.STEM_OUT_CHANNELS
        bneck2_out_channels = cfg.MODEL.MOBILENETS.BNECK2_OUT_CHANNELS
        layers = [
            [bneck2_in_channels, bneck2_in_channels, 3, 1, "RE", False, 16, 1],
            [bneck2_in_channels, bneck2_out_channels, 3, 2, "RE", False, 64, 1],
            [bneck2_out_channels, bneck2_out_channels, 3, 1, "RE", False, 72, 1],
            [bneck2_out_channels, 40, 5, 2, "RE", True, 72, 2],
            [40, 40, 5, 1, "RE", True, 120, 2],
            [40, 40, 5, 1, "RE", True, 120, 2],
            [40, 80, 3, 2, "HS", False, 240, 3],
            [80, 80, 3, 1, "HS", False, 200, 3],
            [80, 80, 3, 1, "HS", False, 184, 3],
            [80, 80, 3, 1, "HS", False, 184, 3],
            [80, 112, 3, 1, "HS", True, 480, 4],
            [112, 112, 3, 1, "HS", True, 672, 4],
            [112, 160, 5, 1, "HS", True, 672, 4],
            [160, 160, 5, 2, "HS", True, 672, 4],
            [160, 160, 5, 1, "HS", True, 960, 4],
        ]
        indices = np.array(layers)[:, -1].tolist()
        r = Counter(indices)
        self.stages = []
        self.init_conv = nn.Sequential(
            Conv2d(in_channels=3, out_channels=bneck2_in_channels, kernel_size=3, stride=2, padding=1),
            FrozenBatchNorm2d(bneck2_in_channels),
            h_swish(inplace=True)
        )

        counts = [0, 3, 6, 10]
        for index in r.keys():
            blocks = []
            name = "layer{}".format(index)
            for _ in range(r[index]):
                in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size = layers[counts[int(
                    index) - 1] + _][:7]
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                blocks.append(MobileBlock(in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size))
            self.add_module(name, nn.Sequential(*blocks))
            self.stages.append(name)

        # self.block = []
        # for in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size, index in layers:
        #     in_channels = _make_divisible(in_channels * multiplier)
        #     out_channels = _make_divisible(out_channels * multiplier)
        #     exp_size = _make_divisible(exp_size * multiplier)
        #     self.block.append(MobileBlock(in_channels, out_channels, kernel_size, stride, nonlinear, se, exp_size))
        # self.block = nn.Sequential(*self.block)

    def forward(self, x):
        outputs = []
        out = self.init_conv(x)
        for stage_name in self.stages:
            out = getattr(self, stage_name)(out)
            outputs.append(out)
        return outputs
