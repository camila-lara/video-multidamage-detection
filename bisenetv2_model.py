# file: bisenetv2_model.py
'''
This script defines the BiSeNetV2 architecture for multiclass semantic segmentation.
It includes the detail branch, semantic branch, bilateral guided aggregation layer,
and segmentation heads. The model takes an input image and produces pixel-wise class
predictions for the configured number of damage categories.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseConvBN(nn.Module):
    def __init__(self, channels, ks=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=stride, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.block(x)


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, ks=3, stride=2, padding=1)

        self.left = nn.Sequential(
            ConvBNReLU(out_ch, out_ch // 2, ks=1, stride=1, padding=0),
            ConvBNReLU(out_ch // 2, out_ch, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_ch * 2, out_ch, ks=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        x = self.fuse(x)
        return x


class GELayerS1(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio=6):
        super().__init__()
        mid_ch = in_ch * exp_ratio

        self.conv1 = ConvBNReLU(in_ch, mid_ch, ks=3, stride=1, padding=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        out = self.relu(out + shortcut)
        return out


class GELayerS2(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio=6):
        super().__init__()
        mid_ch = in_ch * exp_ratio

        self.conv1 = ConvBNReLU(in_ch, mid_ch, ks=3, stride=1, padding=1)

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv1(out)
        out = self.dwconv2(out)
        out = self.conv2(out)

        shortcut = self.shortcut(x)
        out = self.relu(out + shortcut)
        return out


class CEBlock(nn.Module):
    def __init__(self, in_ch=128, out_ch=128):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv_gap = ConvBNReLU(in_ch, in_ch, ks=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(in_ch, out_ch, ks=3, stride=1, padding=1)

    def forward(self, x):
        gap = torch.mean(x, dim=(2, 3), keepdim=True)
        gap = self.bn(gap)
        gap = self.conv_gap(gap)
        out = x + gap
        out = self.conv_last(out)
        return out


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.s2 = nn.Sequential(
            ConvBNReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, ks=3, stride=2, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock(3, 16)

        self.s3 = nn.Sequential(
            GELayerS2(16, 32, exp_ratio=6),
            GELayerS1(32, 32, exp_ratio=6),
        )

        self.s4 = nn.Sequential(
            GELayerS2(32, 64, exp_ratio=6),
            GELayerS1(64, 64, exp_ratio=6),
        )

        self.s5_4 = nn.Sequential(
            GELayerS2(64, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
        )

        self.s5_5 = CEBlock(128, 128)

    def forward(self, x):
        feat2 = self.stem(x)       # 1/4
        feat3 = self.s3(feat2)     # 1/8
        feat4 = self.s4(feat3)     # 1/16
        feat5_4 = self.s5_4(feat4) # 1/32
        feat5_5 = self.s5_5(feat5_4)
        return feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.detail_keep = nn.Sequential(
            DepthwiseConvBN(channels, ks=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.detail_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.semantic_keep = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.semantic_up = nn.Sequential(
            DepthwiseConvBN(channels, ks=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.conv = ConvBNReLU(channels, channels, ks=3, stride=1, padding=1)

    def forward(self, detail, semantic):
        detail_keep = self.detail_keep(detail)
        detail_down = self.detail_down(detail)

        semantic_keep = self.semantic_keep(semantic)
        semantic_up = self.semantic_up(semantic)

        semantic_keep = torch.sigmoid(
            F.interpolate(semantic_keep, size=detail.shape[2:], mode="bilinear", align_corners=False)
        )
        left = detail_keep * semantic_keep

        right = detail_down * torch.sigmoid(semantic_up)
        right = F.interpolate(right, size=detail.shape[2:], mode="bilinear", align_corners=False)

        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes, up_factor=8):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch, ks=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Conv2d(mid_ch, n_classes, kernel_size=1)
        self.up_factor = up_factor

    def forward(self, x, out_size=None):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.cls(x)

        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        elif self.up_factor is not None and self.up_factor > 1:
            x = F.interpolate(x, scale_factor=self.up_factor, mode="bilinear", align_corners=False)

        return x


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes=4, aux_heads=True):
        super().__init__()
        self.aux_heads = aux_heads

        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga = BGALayer(channels=128)

        self.head = SegmentHead(128, 256, n_classes, up_factor=None)

        if self.aux_heads:
            self.aux3 = SegmentHead(32, 64, n_classes, up_factor=None)
            self.aux4 = SegmentHead(64, 64, n_classes, up_factor=None)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=None)
            self.aux5_5 = SegmentHead(128, 128, n_classes, up_factor=None)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out_size = x.shape[2:]

        feat_detail = self.detail(x)
        feat3, feat4, feat5_4, feat5_5 = self.semantic(x)

        feat_head = self.bga(feat_detail, feat5_5)
        logits = self.head(feat_head, out_size=out_size)

        if self.aux_heads and self.training:
            aux3 = self.aux3(feat3, out_size=out_size)
            aux4 = self.aux4(feat4, out_size=out_size)
            aux5_4 = self.aux5_4(feat5_4, out_size=out_size)
            aux5_5 = self.aux5_5(feat5_5, out_size=out_size)
            return logits, aux3, aux4, aux5_4, aux5_5

        return logits