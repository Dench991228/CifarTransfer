from pytorch_pretrained_vit import ViT
import torch.nn as nn
import torch.nn.functional as F


class TVit(ViT):
    def __init__(self, backbone_embedding, count_classes):
        super().__init__()
        self.linear = nn.Linear(backbone_embedding, count_classes)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x
