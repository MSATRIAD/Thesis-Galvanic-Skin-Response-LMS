"""Model exp01: 2D CNN untuk heatmap GLCT (1x128x128 -> biner)."""

import torch.nn as nn


class CognitiveLoadCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()

        def block(cin, cout, pool):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                pool,
            )

        self.block1 = block(1, 32, nn.MaxPool2d(2, 2))     # 128->64
        self.block2 = block(32, 64, nn.MaxPool2d(2, 2))    # 64->32
        self.block3 = block(64, 128, nn.MaxPool2d(2, 2))   # 32->16
        self.block4 = block(128, 256, nn.AdaptiveAvgPool2d((4, 4)))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.classifier(x)
