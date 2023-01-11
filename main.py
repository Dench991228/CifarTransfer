from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn
from loaders import get_train_loader, get_test_loader
from TransferVit import TVit
import argparse
import torch.optim


args = argparse.ArgumentParser(description="Transferring ViT to CIFAR10 or CIFAR100")
args.add_argument("--dataset", default='cifar10')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, criterion, optimizer, train_loader):
    avg_loss = 0
    count_correct = 0
    count_wrong = 0
    model.train()
    for idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        # 得到图片的分类归一化概率
        print(images.shape)
        outputs = model(images)
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


def test(model, criterion, test_loader):
    loss = 0
    correct = 0
    wrong = 0
    model.eval()
    for idx, (images, targets) in enumerate(test_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
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
    learning_rate = 1e-2
    transfer_ratio = 1e-2
    model = TVit('B_16', img_size=224, backbone_embedding=768, count_classes=10)
    model.to(device)
    print("pretrained model loaded")
    state_dict = model.state_dict()
    del state_dict['classifier.weight']
    fine_tuned_params = []
    for k in state_dict:
        fine_tuned_params.append(model.get_parameter(k))
    parameters = [{'params': fine_tuned_params, 'lr': learning_rate * transfer_ratio},
                  {'params': [model.classifier.weight], 'lr': learning_rate}]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=2e-4, T_max=60)
    criterion = nn.CrossEntropyLoss()
    for id in range(60):
        print(f"Epoch {id}")
        train_one_epoch(model, criterion, optimizer, get_train_loader(args.dataset))
        test(model, criterion, get_test_loader(args.dataset))
