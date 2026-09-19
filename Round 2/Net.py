import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Net(nn.Module):
    def __init__(self, num_classes=72, dropout=0.3):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # ตัด maxpool ออก: input เล็ก (64x64) ถ้า downsample ตาม conv1+maxpool+4 stage
        # แบบเดิมจะยุบเหลือ 2x2 ก่อนถึง fc รายละเอียดตัวอักษรหายไปเยอะ
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
