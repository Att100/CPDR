import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.blocks import Block


class Decoder:
    modules = dict()

    @staticmethod
    def register(classobj):
        Decoder.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Decoder.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Decoder.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Decoder.get(name)(*args, **kwargs)


@Decoder.register
class DecoderA(nn.Layer):
    name = "decoder_a"

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_in = Block.make("conv2d_bn_relu", in_channels, in_channels, 3, 1)
        self.conv_out = Block.make("conv2d_bn_relu", in_channels, out_channels, 1)

    def forward(self, x, res):
        out = self.conv_in(
            F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True) + \
            res)
        out = self.conv_out(out)
        return out, F.upsample(out, scale_factor=2, mode='bilinear', align_corners=True)
    

@Decoder.register
class DecoderB(nn.Layer):
    name = "decoder_b"

    def __init__(self, in_channels, out_channels, reduction=2, proj_reduction=1):
        super().__init__()

        mid_channels = in_channels // reduction
        proj_channels = out_channels // proj_reduction
        
        self.conv_in = Block.make("conv2d_bn_relu", in_channels, mid_channels, 3, 1)
        self.conv_out = Block.make("conv2d_bn_relu", mid_channels*2, out_channels, 1)
        self.conv_proj = Block.make("conv2d_bn_relu", out_channels, proj_channels, 1)

    def forward(self, x, res):
        out = self.conv_in(
            F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True))
        out = paddle.concat([out, res], axis=1)
        out = self.conv_out(out)
        return out, F.upsample(
            self.conv_proj(out), scale_factor=2, mode='bilinear', align_corners=True)


@Decoder.register
class DecoderC(nn.Layer):
    name = "decoder_c"

    def __init__(self, in_channels, out_channels, proj_channels):
        super().__init__()

        self.conv_in = Block.make("conv2d_bn_relu", in_channels, in_channels, 3, 1)
        self.conv_out = Block.make("conv2d_bn_relu", in_channels, out_channels, 1)
        self.conv_proj = Block.make("conv2d_bn_relu", out_channels, proj_channels, 1)

    def forward(self, x, res):
        out = self.conv_in(
            F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True) + \
            res)
        out = self.conv_out(out)
        return out, F.upsample(
            self.conv_proj(out), scale_factor=2, mode='bilinear', align_corners=True)
        

@Decoder.register
class DecoderD(nn.Layer):
    name = "decoder_d"

    def __init__(self, in_channels, out_channels, proj_channels, reduction=2):
        super().__init__()

        mid_channels = in_channels // reduction
        
        self.conv_in = Block.make("conv2d_bn_relu", in_channels, mid_channels, 3, 1)
        self.conv_out = Block.make("conv2d_bn_relu", mid_channels*2, out_channels, 1)
        self.conv_proj = Block.make("conv2d_bn_relu", out_channels, proj_channels, 1)

    def forward(self, x, res):
        out = self.conv_in(
            F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True))
        out = paddle.concat([out, res], axis=1)
        out = self.conv_out(out)
        return out, F.upsample(
            self.conv_proj(out), scale_factor=2, mode='bilinear', align_corners=True)
    
