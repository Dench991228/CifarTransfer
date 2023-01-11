from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn
from loaders import get_train_loader, get_test_loader
from TransferVit import TVit
import argparse
import torch.optim


args = argparse.ArgumentParser(description="Transferring ViT to CIFAR10 or CIFAR100")
args.add_argument("--dataset", default='cifar10')
model = ViT('B_16_imagenet1k', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 把后面的分类头去掉，换成我自己的分类头
backboneViT = nn.Sequential(*list(model.children())[:-1])
backboneViT.to(device)
print("Backbone Vision Transformer prepared")


def train_one_epoch(backbone, head, criterion, optimizer, train_loader):
    avg_loss = 0
    count_correct = 0
    count_wrong = 0
    backbone.train()
    head.train()
    for idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        # 先输出得到backbone部分的嵌入式表达
        outputs = backbone(images)[:, 0]
        # 然后是分类器的表达
        outputs = head(outputs)
        # 得到损失函数
        loss = criterion(outputs, targets)
        avg_loss += loss
        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算正确率
        predicts = torch.argmax(outputs, dim=1, keepdim=False)
        count_correct += ((predicts == targets).sum())
        count_wrong += ((predicts != targets).sum())
    avg_loss /= len(train_loader)
    avg_correct = count_correct / (count_wrong + count_correct)
    print(f"train loss: {avg_loss}")
    print(f"train acc: {avg_correct}")


def test(backbone, head, criterion, test_loader):
    loss = 0
    correct = 0
    wrong = 0
    backbone.eval()
    head.eval()
    for idx, (images, targets) in enumerate(test_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = backbone(images)[:, 0]
        outputs = head(outputs)
        batch_loss = criterion(outputs)
        loss += batch_loss
        predicts = torch.argmax(outputs, dim=1)
        correct += ((predicts == targets).sum())
        wrong += ((predicts == targets).sum())
    avg_loss = loss/len(test_loader)
    avg_correct = correct / (wrong+correct)
    print(f"test loss: {avg_loss}")
    print(f"test acc: {avg_correct}")


if __name__ == '__main__':
    args = args.parse_args()
    train_loader = get_train_loader(args.dataset)
    test_loader = get_test_loader(args.dataset)
    classifier_head = TVit(768, 10 if args.dataset == 'cifar10' else 100)
    classifier_head.to(device)
    learning_rate = 1e-2
    transfer_ratio = 1e-2
    parameters = [{'params': backboneViT.parameters(), 'lr': learning_rate * transfer_ratio},
                  {'params': classifier_head.parameters()}]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=2e-4, T_max=60)
    criterion = nn.CrossEntropyLoss()
    for id in range(60):
        print(f"Epoch {id}")
        train_one_epoch(backboneViT, classifier_head, criterion, optimizer, get_train_loader(args.dataset))
        test(backboneViT, classifier_head, criterion, get_test_loader(args.dataset))
