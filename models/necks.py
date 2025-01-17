import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.blocks import Block


class Neck:
    modules = dict()

    @staticmethod
    def register(classobj):
        Neck.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Neck.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Neck.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Neck.get(name)(*args, **kwargs)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
@Neck.register
class CPDR_A(nn.Layer):
    name = "cpdr_a"

    def __init__(self, block_cfgs: list):
        super().__init__()

        for name, block_name, params in block_cfgs:
            self.__setattr__(name, Block.make(block_name, *params))
            
    def forward(self, x):
        out_x2, out_x4, out_x8, out = x

        # cpdr stage 1
        out = self.conv(out * self.attn_x8(out_x8))
        out_x8 = self.conv_x8(paddle.concat([self.conv_x8_down(out), out_x8], axis=1) * self.attn_x4(out_x4))
        out_x4 = self.conv_x4(paddle.concat([self.conv_x4_down(out_x8), out_x4], axis=1) * self.attn_x2(out_x2))

        # cpdr stage 2
        sp_w_x8 = F.interpolate(self.attn2_x8(out), scale_factor=0.5, mode='bilinear', align_corners=True)
        sp_w_x4 = F.interpolate(self.attn2_x4(out_x8), scale_factor=0.5, mode='bilinear', align_corners=True)
        out_x4 = self.conv2_x4(out_x4 * sp_w_x4)
        out_x8 = self.conv2_x8(paddle.concat([
            self.conv_x4_up(F.upsample(out_x4, scale_factor=2, mode='bilinear', align_corners=True)),
            out_x8], axis=1) * sp_w_x8)
        out = self.conv2(paddle.concat([
            self.conv_x8_up(F.upsample(out_x8, scale_factor=2, mode='bilinear', align_corners=True)),
            out], axis=1))

        return out_x4, out_x8, out
    

@Neck.register
class CPDR_B(nn.Layer):
    name = "cpdr_b"

    def __init__(self, block_cfgs: list):
        super().__init__()

        for name, block_name, params in block_cfgs:
            self.__setattr__(name, Block.make(block_name, *params))

    def forward(self, x):
        out_x2, out_x4, out_x8, out = x

        # cpdr two stage in one 
        outputs = []
        low_feat = out_x2
        for i, high_feat in enumerate([out_x4, out_x8, out]):
            low_feat = self.__getattr__(f"dacf{i+1}")(low_feat, high_feat)
            outputs.append(low_feat)

        return tuple(outputs)