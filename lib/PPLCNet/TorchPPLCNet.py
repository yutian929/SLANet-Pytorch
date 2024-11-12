import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

MODEL_URLS = {
    "PPLCNet_x0.25": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    # Add other URLs as needed
}

NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hardsigmoid(x)
        return identity * x

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size=3, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(in_channels, in_channels, dw_size, stride, groups=in_channels)
        if use_se:
            self.se = SEModule(in_channels)
        self.pw_conv = ConvBNLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

class PPLCNet(nn.Module):
    def __init__(self, in_channels=3, scale=1.0, pretrained=False):
        super(PPLCNet, self).__init__()
        self.scale = scale
        self.out_channels = [
            int(NET_CONFIG["blocks3"][-1][2] * scale),
            int(NET_CONFIG["blocks4"][-1][2] * scale),
            int(NET_CONFIG["blocks5"][-1][2] * scale),
            int(NET_CONFIG["blocks6"][-1][2] * scale),
        ]

        self.conv1 = ConvBNLayer(in_channels, make_divisible(16 * scale), 3, 2)

        self.blocks2 = self._make_layer(NET_CONFIG["blocks2"], scale)
        self.blocks3 = self._make_layer(NET_CONFIG["blocks3"], scale)
        self.blocks4 = self._make_layer(NET_CONFIG["blocks4"], scale)
        self.blocks5 = self._make_layer(NET_CONFIG["blocks5"], scale)
        self.blocks6 = self._make_layer(NET_CONFIG["blocks6"], scale)

        if pretrained:
            self._load_pretrained(MODEL_URLS["PPLCNet_x{}".format(scale)])

    def _make_layer(self, config, scale):
        layers = []
        for k, in_c, out_c, s, se in config:
            layers.append(
                DepthwiseSeparable(
                    make_divisible(in_c * scale),
                    make_divisible(out_c * scale),
                    s,
                    k,
                    se
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)
        return outs

    def _load_pretrained(self, pretrained_url):
        state_dict = load_state_dict_from_url(pretrained_url, progress=True)
        self.load_state_dict(state_dict)
