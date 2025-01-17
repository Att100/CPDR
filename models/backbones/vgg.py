import paddle
import paddle.nn as nn


cfgs = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG16(nn.Layer):
    def __init__(self, pretrained=False, ckpt_path=None):
        super().__init__()
        self.features = self._make_layer('D', cfgs)
        
        """
        self.avgpool = nn.AdaptiveAvgPool2D((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        """

        if pretrained and ckpt_path is not None:
            self.set_state_dict(paddle.load(ckpt_path))
            print(ckpt_path, "loaded")

        conv_children = list(self.children())[0]

        self.conv_block1 = nn.Sequential(conv_children[:7])
        self.conv_block2 = nn.Sequential(conv_children[7:14])
        self.conv_block3 = nn.Sequential(conv_children[14:24])
        self.conv_block4 = nn.Sequential(conv_children[24:34])
        self.conv_block5 = nn.Sequential(conv_children[34:])

    def forward(self, x):
        """
        x = self.features(x)
        x = self.avgpool(x)

        x = paddle.flatten(x, 1)
        out = self.classifier(x)

        return out
        """

        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(feat1)
        feat3 = self.conv_block3(feat2)
        feat4 = self.conv_block4(feat3)
        out = self.conv_block5(feat4)

        return feat1, feat2, feat3, feat4, out

    def _make_layer(self, key, cfgs):
        layers = []
        in_channels = 3
        for v in cfgs[key]:
            if v == 'M':
                layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2D(in_channels, v, 3, padding=1),
                    nn.BatchNorm2D(v),
                    nn.ReLU()
                ]
                in_channels = v
        return nn.Sequential(*layers)
    