from torch import nn
from collections import OrderedDict

class SSDBackbone(nn.Module):
    def __init__(self, backbone, backbone_out_channels=512):
        super().__init__()
        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        layer1 = backbone.layer1
        layer2 = backbone.layer2
        layer3 = backbone.layer3
        layer4 = backbone.layer4

        self.features = nn.Sequential(layer0, layer1, layer2, layer3)
        self.upsampling= nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels//2, out_channels=backbone_out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.extra = nn.ModuleList(
            [
                nn.Sequential(
                    layer4,
                    nn.Conv2d(in_channels=backbone_out_channels, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=4),
                    nn.ReLU(inplace=True),
                )
            ]
        )


    def forward(self, x):
        x = self.features(x)
        output = [self.upsampling(x)]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])