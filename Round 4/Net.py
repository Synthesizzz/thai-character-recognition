import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Net(nn.Module):
    def __init__(self, num_classes=72, dropout=0.3, use_maxpool=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # use_maxpool=False ตัด maxpool ออก (เหมาะกับ input เล็ก เช่น 64 ที่ไม่อยากให้ feature map ยุบเกินไป)
        # ค่านี้ต้องตรงกันระหว่างตอนเทรนกับ inference (บันทึกไว้ใน norm_stats.json)
        if not use_maxpool:
            self.backbone.maxpool = nn.Identity()
        # default = ไม่ตัด maxpool (แบบ round 3): input 224x224 ตรงกับที่ ResNet18 pretrained มา
        # downsample เต็ม path (conv1+maxpool+4 stage) ได้ feature map 7x7 ก่อน avgpool ตามมาตรฐาน ไม่ยุบเกินไป
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
