import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class DynamicCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["channels"]
        num_blocks = config["num_blocks"]

        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=False)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = self.act(feat)
        feat = self.body(feat)
        out = self.conv_last(feat)
        return x + out


def build_model(config):
    return DynamicCNN(config)