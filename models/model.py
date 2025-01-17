import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.blocks import Block
from models.decoders import Decoder
from models.necks import Neck
from models.backbones import Backbone


class Model:
    modules = dict()

    @staticmethod
    def register(classobj):
        Model.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Model.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Model.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Model.get(name)(*args, **kwargs)


@Model.register
class FPN(nn.Layer):
    name = "fpn"

    def __init__(self, structure_cfg: dict, pretrained: bool, backbone_ckpt_path: str=None):
        super().__init__()

        # encoder/backbone
        self.backbone = Backbone.make(structure_cfg.get('backbone').get('select'), pretrained, backbone_ckpt_path)

        # mappers
        self.n_mappers = 0
        for [name, block_name, params] in structure_cfg.get('mappers').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))
            self.n_mappers += 1

        # decoders
        for [name, block_name, params] in structure_cfg.get('decoders').get('block_cfgs'):
            self.__setattr__(name, Decoder.make(block_name, *params))

        # heads
        for [name, block_name, params] in structure_cfg.get('heads').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))

    def forward(self, x):
        # backbone
        # output -> [feat1, feat2, feat3, feat4, output]
        feats = self.backbone(x)

        # mappers
        # output -> [output, feat4, feat3, feat2, feat1]
        if self.n_mappers > 0:
            reduced_feats = []
            for i, feat in enumerate(list(feats)[::-1]):
                reduced_feats.append(self.__getattr__('mapr{}'.format(i+1))(feat))
        # without mappers
        else:
            reduced_feats = feats[::-1]

        # decoders
        # output -> [out_x2, out_x4, out_x8, out]
        decoded_feats = []
        _output = reduced_feats[0]
        for i, feat in enumerate(reduced_feats[1:]):
            _output, output = self.__getattr__('dec{}'.format(i+1))(_output, feat)
            decoded_feats.append(output)
        
        # heads
        # output -> [out, out_x8, out_x4, ...]
        preds = []
        for i, feat in enumerate(list(decoded_feats)[::-1][:3]):
            preds.append(self.__getattr__('classifier{}'.format(i+1))(feat))

        # out, out_x8, out_x4
        return tuple(preds)
    

@Model.register
class FPN_With_Neck(nn.Layer):
    name = "fpn_with_neck"

    def __init__(self, structure_cfg: dict, pretrained: bool, backbone_ckpt_path: str=None):
        super().__init__()

        # encoder/backbone
        self.backbone = Backbone.make(structure_cfg.get('backbone').get('select'), pretrained, backbone_ckpt_path)

        # mappers
        self.n_mappers = 0
        for [name, block_name, params] in structure_cfg.get('mappers').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))
            self.n_mappers += 1

        # decoders
        for [name, block_name, params] in structure_cfg.get('decoders').get('block_cfgs'):
            self.__setattr__(name, Decoder.make(block_name, *params))

        # neck
        self.neck = Neck.make(
            structure_cfg.get('neck').get('select'),
            structure_cfg.get('neck').get('block_cfgs'))

        # heads
        for [name, block_name, params] in structure_cfg.get('heads').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))
            
        self.return_decoded_features = structure_cfg.get('return_decoded_features', False)

    def forward(self, x):
        # backbone
        # output -> [feat1, feat2, feat3, feat4, output]
        feats = self.backbone(x)

        # mappers
        # output -> [output, feat4, feat3, feat2, feat1]
        if self.n_mappers > 0:
            reduced_feats = []
            for i, feat in enumerate(list(feats)[::-1]):
                reduced_feats.append(self.__getattr__('mapr{}'.format(i+1))(feat))
        # without mappers
        else:
            reduced_feats = feats[::-1]

        # decoders
        # output -> [out_x2, out_x4, out_x8, out]
        decoded_feats = []
        _output = reduced_feats[0]
        for i, feat in enumerate(reduced_feats[1:]):
            _output, output = self.__getattr__('dec{}'.format(i+1))(_output, feat)
            decoded_feats.append(output)
        
        # neck
        # output -> [..., out_x4, out_x8, out]
        neck_feats = self.neck(tuple(decoded_feats))

        # heads
        # output -> [out, out_x8, out_x4, ...]
        preds = []
        for i, feat in enumerate(list(neck_feats)[::-1]):
            preds.append(self.__getattr__('classifier{}'.format(i+1))(feat))

        if self.return_decoded_features:
            return decoded_feats[::-1], tuple(preds)
        
        # out, out_x8, out_x4
        return tuple(preds)





################ VIS ################
@Model.register
class FPNVis(nn.Layer):
    name = "fpn_vis"

    def __init__(self, structure_cfg: dict, pretrained: bool, backbone_ckpt_path: str=None):
        super().__init__()

        # encoder/backbone
        self.backbone = Backbone.make(structure_cfg.get('backbone').get('select'), pretrained, backbone_ckpt_path)

        # mappers
        self.n_mappers = 0
        for [name, block_name, params] in structure_cfg.get('mappers').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))
            self.n_mappers += 1

        # decoders
        for [name, block_name, params] in structure_cfg.get('decoders').get('block_cfgs'):
            self.__setattr__(name, Decoder.make(block_name, *params))

        # heads
        for [name, block_name, params] in structure_cfg.get('heads').get('block_cfgs'):
            self.__setattr__(name, Block.make(block_name, *params))
            
        self.activations = []
        self.gradients = []

    def forward(self, x):
        # backbone
        # output -> [feat1, feat2, feat3, feat4, output]
        feats = self.backbone(x)

        # mappers
        # output -> [output, feat4, feat3, feat2, feat1]
        if self.n_mappers > 0:
            reduced_feats = []
            for i, feat in enumerate(list(feats)[::-1]):
                reduced_feats.append(self.__getattr__('mapr{}'.format(i+1))(feat))
        # without mappers
        else:
            reduced_feats = feats[::-1]

        # decoders
        # output -> [out_x2, out_x4, out_x8, out]
        decoded_feats = []
        _output = reduced_feats[0]
        for i, feat in enumerate(reduced_feats[1:]):
            _output, output = self.__getattr__('dec{}'.format(i+1))(_output, feat)
            output.register_hook(self._save_gradient)
            decoded_feats.append(output)
        self.activations = decoded_feats
        
        # heads
        # output -> [out, out_x8, out_x4, ...]
        preds = []
        for i, feat in enumerate(list(decoded_feats)[::-1][:3]):
            preds.append(self.__getattr__('classifier{}'.format(i+1))(feat))

        # out, out_x8, out_x4
        return tuple(preds)
    
    def _save_gradient(self, grad):
        self.gradients.append(grad)
#####################################