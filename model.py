import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

model_list = ['swin_v2_t', 'swin_t', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_x_3_2gf', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'resnext50_32x4d', 'resnet50', 'efficientnet_b0']

class BaseModel(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = torchvision.models.get_model(backbone, pretrained=True)
        
        if backbone not in model_list:
            raise ValueError(f"backbone model name is wrong!")
        
        if 'swin' in backbone:
            self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
        elif 'efficientnet' in backbone:
            self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
        else:
            self.backbone.fn = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
