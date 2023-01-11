from pytorch_pretrained_vit import ViT
import torch.nn as nn
import torch.nn.functional as F
import torch


class TVit(ViT):
    def __init__(self, name, img_size, backbone_embedding, count_classes):
        super().__init__(name, pretrained=True, image_size=img_size)  # 先初始化一个骨干模型
        self.classifier = nn.Linear(backbone_embedding, count_classes, bias=False)

    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        x = self.norm(x)[:, 0]  # b,d
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
