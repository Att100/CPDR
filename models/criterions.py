import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Criterion:
    modules = dict()

    @staticmethod
    def register(classobj):
        Criterion.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        Criterion.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return Criterion.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return Criterion.get(name)(*args, **kwargs)
    

def _iou_loss(pred, target, smooth=1):
    intersection = paddle.sum(target * pred, axis=[1,2,3])
    union = paddle.sum(target, axis=[1,2,3]) + paddle.sum(pred, axis=[1,2,3]) - intersection
    iou = paddle.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou

def _dice_loss(pred, target, eps=1e-5):
    intersection = paddle.sum(target * pred, axis=[1,2,3])
    dice = (2*intersection + eps) / \
        (paddle.sum(target, axis=[1,2,3]) + paddle.sum(pred, axis=[1,2,3]) + eps)
    return paddle.mean(1 - dice)


@Criterion.register
class BCEWithLogitsLoss(nn.Layer):
    name = "bce_logits"

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = F.binary_cross_entropy(F.sigmoid(pred[0]), target)
        return loss
    
@Criterion.register
class MsDiceLoss(nn.Layer):
    name = "ms_dice_loss"
    
    def __init__(self, w1=[1, 0.8, 0.5]):
        super().__init__()
        self.w1 = w1
        
    def forward(self, pred, target):
        out, x8_out, x4_out = tuple(pred)
        out, x8_out, x4_out = F.sigmoid(out), F.sigmoid(x8_out), F.sigmoid(x4_out)
        
        target = target.unsqueeze(1)
        target_4x = F.interpolate(
            target, x4_out.shape[2:], mode='bilinear', align_corners=True)
        target_8x = F.interpolate(
            target, x8_out.shape[2:], mode='bilinear', align_corners=True)
        
        loss = self.w1[0] * _dice_loss(out, target) + \
            self.w1[1] * _dice_loss(x8_out, target_8x) + \
            self.w1[2] * _dice_loss(x4_out, target_4x)
        return loss

@Criterion.register
class BCEAndIOUWithLogitsLoss(nn.Layer):
    name = "bce_iou_logits"

    def __init__(self, w1=1):
        super().__init__()
        self.w1 = w1

    def bce_iou_loss(self, pred, target):
        loss = F.binary_cross_entropy(F.sigmoid(pred[0]), target) + \
            self.w1 * _iou_loss(F.sigmoid(pred[0]), target)
        return loss


@Criterion.register
class MsBCEWithLogitsLoss(nn.Layer):
    name = "ms_bce_logits"

    def __init__(self, w1=[1, 0.8, 0.5]):
        super().__init__()
        self.w1 = w1

    def forward(self, pred, target):
        out, x8_out, x4_out = tuple(pred)
        target = target.unsqueeze(1)
        target_4x = F.interpolate(
            target, x4_out.shape[2:], mode='bilinear', align_corners=True)
        target_8x = F.interpolate(
            target, x8_out.shape[2:], mode='bilinear', align_corners=True)
        
        _loss = F.binary_cross_entropy(F.sigmoid(out), target)
        _4x_loss = F.binary_cross_entropy(F.sigmoid(x4_out), target_4x)
        _8x_loss = F.binary_cross_entropy(F.sigmoid(x8_out), target_8x)

        loss = self.w1[0] * _loss + self.w1[1] * _8x_loss + self.w1[2] * _4x_loss
        return loss


@Criterion.register
class MsBCEAndIOUWithLogitsLoss(nn.Layer):
    name = "ms_bce_iou_logits"

    def __init__(self, w1=[1, 0.8, 0.5], w2=[1, 1, 1]):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred, target):
        out, x8_out, x4_out = tuple(pred)
        target = target.unsqueeze(1)
        target_4x = F.interpolate(
            target, x4_out.shape[2:], mode='bilinear', align_corners=True)
        target_8x = F.interpolate(
            target, x8_out.shape[2:], mode='bilinear', align_corners=True)
        
        _loss = F.binary_cross_entropy(F.sigmoid(out), target) + \
            self.w2[0] * _iou_loss(F.sigmoid(out), target)
        _4x_loss = F.binary_cross_entropy(F.sigmoid(x4_out), target_4x) + \
            self.w2[1] * _iou_loss(F.sigmoid(x4_out), target_4x)
        _8x_loss = F.binary_cross_entropy(F.sigmoid(x8_out), target_8x) + \
            self.w2[2] * _iou_loss(F.sigmoid(x8_out), target_8x)

        loss = self.w1[0] * _loss + self.w1[1] * _8x_loss + self.w1[2] * _4x_loss
        return loss



