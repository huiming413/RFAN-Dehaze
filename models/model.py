import torch.nn as nn
import torch.nn.functional as F
from .deconv import FastDeconv
import torch


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True,
               groups=1):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias,
                     groups=groups)


class eca_layer(nn.Module):
    """
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Block(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.LeakyReLU(0.05, True)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + x
        return x


class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RefineBlock(nn.Module):
    def __init__(self, dim, esa_channels=32):
        super().__init__()
        self.rfblock = Block(dim)
        self.esa = ESA(esa_channels, dim, nn.Conv2d)
        self.c1 = conv_layer(dim, dim, 1)
        self.eca = eca_layer(dim, k_size=5)

    def forward(self, x):
        x1 = self.rfblock(x)
        x2 = self.rfblock(x1)
        x3 = self.rfblock(x2)
        out = x + x3
        out = self.eca(out)
        out = self.esa(self.c1(out))
        return out


class RFABS(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52):
        super(RFABS, self).__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, kernel_size=3)

        self.block1 = RefineBlock(feature_channels)
        self.block2 = RefineBlock(feature_channels)
        self.block3 = RefineBlock(feature_channels)
        self.block4 = RefineBlock(feature_channels)
        self.block5 = RefineBlock(feature_channels)
        self.block6 = RefineBlock(feature_channels)

        self.conv_2 = conv_layer(feature_channels, out_channels, kernel_size=3)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block1(out_feature)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)
        out_low_resolution = out_b6 + out_feature
        output = self.conv_2(out_low_resolution)
        return output


class Dehaze(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### RFAB blocks
        self.block = RFABS(ngf * 4, ngf * 4, ngf * 4)

        ###### upsample 
        self.up1 = nn.Sequential(conv_layer(ngf * 4, ngf * 8, 1), nn.PixelShuffle(2), nn.ReLU(True))
        self.up2 = nn.Sequential(conv_layer(ngf * 2, ngf * 4, 1), nn.PixelShuffle(2), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        x_deconv = self.deconv(input)  # preprocess

        x_down1 = self.down1(x_deconv)  # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)  # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)  # [bs, 256, 64, 64]

        x = self.block(x_down3)
        x_up1 = self.up1(x)  # [bs, 128, 128, 128]
        x_up2 = self.up2(x_up1 + x_down2)  # [bs, 64, 256, 256]
        out = self.up3(x_up2)  # [bs,  3, 256, 256]
        return out
