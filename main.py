from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn

model = ViT('B_16_imagenet1k', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 把后面的分类头去掉，换成我自己的分类头
backboneViT = nn.Sequential(*list(model.children())[:-1])
backboneViT.to(device)