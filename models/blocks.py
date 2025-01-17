import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.models.mobilenetv2 import InvertedResidual
from paddle.vision.models.resnet import BottleneckBlock


class Block:
    modules = dict()

    @staticmethod
    def register(classobj):
        Block.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Block.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Block.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Block.get(name)(*args, **kwargs)


## Register Paddle layers ##
Block.register_with_name(nn.Conv2D, "conv2d")
Block.register_with_name(InvertedResidual, "inverted_residual")
############################


@Block.register
class Conv2dBnReLU(nn.Layer):
    name = "conv2d_bn_relu"

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        return F.relu(out)
    
    
@Block.register
class ResidualBlock(nn.Layer):
    name = "residual_block"
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        downsample = None
        
        assert out_channels % BottleneckBlock.expansion == 0
        channels = out_channels // BottleneckBlock.expansion
        
        if stride != 1 or in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    channels * BottleneckBlock.expansion,
                    1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(channels * BottleneckBlock.expansion),
            )
        self.bottleneck = BottleneckBlock(
            in_channels, channels, stride, downsample,
            norm_layer=nn.BatchNorm2D)
        
    def forward(self, x):
        return self.bottleneck(x)
    

@Block.register
class SE(nn.Layer):
    """
    SE Channel Attention Module
    """

    name = "se"

    def __init__(self, in_channels, ratio=16, mult=1):
        super().__init__()

        self.mult = mult

        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_channels, in_channels//ratio, 1, bias_attr=False)
        self.fc2 = nn.Conv2D(in_channels//ratio, in_channels, 1, bias_attr=False)
        
    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avgpool(x))))
        return F.sigmoid(avg_out * self.mult)


@Block.register
class SpatialAttention(nn.Layer):
    """
    CBAM Spatial Attention Module
    """

    name = "cbam-sp"

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size should be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, keepdim=True, axis=1)
        out = paddle.concat([avg_out, max_out], axis=1)
        out = self.conv(out)
        return F.sigmoid(out)


@Block.register
class DACF(nn.Layer):
    """
    DACF: Dual Attentional Cross Fusion
    """
    
    name = "dacf"
    
    def __init__(self, channels, se_ratio=2, sp_ks=7):
        super().__init__()
        
        self.se = Block.make("se", *(channels, se_ratio, 2))
        self.sa = Block.make("cbam-sp", *[sp_ks])
        self.conv_high_reduce = Block.make("conv2d_bn_relu", *(channels, channels//2, 3, 1))
        self.conv_low_reduce = Block.make("conv2d_bn_relu", *(channels, channels//2, 3, 1))
        self.conv_down = Block.make("conv2d_bn_relu", *(channels//2, channels//2, 3, 1, 2))
        self.conv_fuse_reduce = Block.make("conv2d_bn_relu", *(channels, channels//2, 1))
        self.conv_out = Block.make("conv2d_bn_relu", *(channels//2, channels//2, 3, 1))
        
    def forward(self, low_feat, high_feat):
        se_attn = self.se(low_feat)
        sa_attn = F.interpolate(
            self.sa(high_feat), scale_factor=0.5, mode='bilinear', align_corners=True)
        low_refine = self.conv_low_reduce(low_feat * sa_attn)
        high_refine = self.conv_high_reduce(high_feat * se_attn)
        low_cat = paddle.concat([low_refine, self.conv_down(high_refine)], axis=1)
        high_out = high_refine + self.conv_fuse_reduce(F.interpolate(
            low_cat, scale_factor=2, mode='bilinear', align_corners=True))
        return self.conv_out(high_out)


        
    
    
    



    
    
