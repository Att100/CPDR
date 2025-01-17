import paddle
import paddle.nn as nn

from paddle.vision.models import mobilenet_v2
from models.backbones._efficientnet.efficientnet import EfficientNet
from models.backbones.vgg import VGG16
from paddle.vision.models import resnet50


class Backbone:
    modules = dict()

    @staticmethod
    def register(classobj):
        Backbone.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Backbone.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Backbone.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Backbone.get(name)(*args, **kwargs)


Backbone.register_with_name(VGG16, "vgg16")


@Backbone.register
class ResNet50(nn.Layer):
    name = "resnet50"

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = resnet50(pretrained)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)  # (N, 64, H/2, W/2)
        res0 = self.backbone.maxpool(x)

        res1 = self.backbone.layer1(res0)  # (N, 256, H/4, W/4)
        res2 = self.backbone.layer2(res1)  # (N, 512, H/8, W/8)
        res3 = self.backbone.layer3(res2)  # (N, 1024, H/16, W/16)
        res4 = self.backbone.layer4(res3)  # (N, 2048, H/32, W/32)

        return x, res1, res2, res3, res4


@Backbone.register
class MobileNetV2(nn.Layer):
    name = "mobilenet_v2"

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        mobilenet = mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features

    def forward(self, x):
        # stage 1
        feat1 = self.features._sub_layers['0'](x)

        # stage 2
        feat2 = feat1
        for key in ['1', '2', '3']:
            feat2 = self.features[key](feat2)

        # stage 3
        feat3 = feat2
        for key in ['4', '5', '6']:
            feat3 = self.features[key](feat3)

        # stage 4
        feat4 = feat3
        for key in ['7', '8', '9', '10']:
            feat4 = self.features[key](feat4)

        # stage 5
        out = feat4
        for key in ['11', '12', '13', '14', '15', '16', '17', '18']:
            out= self.features[key](out)

        return feat1, feat2, feat3, feat4, out
    

@Backbone.register
class MobileNetV2Lite(nn.Layer):
    name = "mobilenet_v2lite"

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        mobilenet = mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features

    def forward(self, x):
        # stage 1
        feat1 = self.features._sub_layers['0'](x)

        # stage 2
        feat2 = feat1
        for key in ['1', '2', '3']:
            feat2 = self.features[key](feat2)

        # stage 3
        feat3 = feat2
        for key in ['4', '5', '6']:
            feat3 = self.features[key](feat3)

        # stage 4
        feat4 = feat3
        for key in ['7', '8', '9', '10']:
            feat4 = self.features[key](feat4)

        # stage 5
        out = feat4
        for key in ['11', '12', '13', '14', '15', '16']:
            out = self.features[key](out)

        return feat1, feat2, feat3, feat4, out
    

@Backbone.register
class EfficientNetB0(nn.Layer):
    name = 'efficientnet_b0'

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b0', features_only=True)
        if pretrained and ckpt_path is not None:
            self.backbone.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

    def forward(self, x):
        return self.backbone(x)
    

@Backbone.register
class EfficientNetB1(nn.Layer):
    name = 'efficientnet_b1'

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b1', features_only=True)
        if pretrained and ckpt_path is not None:
            self.backbone.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

    def forward(self, x):
        return self.backbone(x)
    

@Backbone.register
class EfficientNetB2(nn.Layer):
    name = 'efficientnet_b2'

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b2', features_only=True)
        if pretrained and ckpt_path is not None:
            self.backbone.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

    def forward(self, x):
        return self.backbone(x)
    
    def load_pretrained(self, path: str):
        self.backbone.set_state_dict(paddle.load(path))
    

@Backbone.register
class EfficientNetB3(nn.Layer):
    name = 'efficientnet_b3'

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b3', features_only=True)
        if pretrained and ckpt_path is not None:
            self.backbone.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

    def forward(self, x):
        return self.backbone(x)


@Backbone.register
class EfficientNetB4(nn.Layer):
    name = 'efficientnet_b4'

    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()

        self.backbone = EfficientNet.from_name('efficientnet-b4', features_only=True)
        if pretrained and ckpt_path is not None:
            self.backbone.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

    def forward(self, x):
        return self.backbone(x)