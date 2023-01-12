import datetime

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
log_file = open("log.txt", 'w')
best_acc = 0

def train_one_epoch(model, criterion, optimizer, train_loader):
    avg_loss = 0
    count_correct = 0
    count_wrong = 0
    model.train()
    for idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        # 得到图片的分类归一化概率
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
        batch_correct = ((predicts == targets).sum())
        batch_wrong = ((predicts != targets).sum())
        count_correct += batch_correct
        count_wrong += batch_wrong
        log_file.write(f"Batch {idx}, correct {batch_correct}, wrong {batch_wrong}, loss {loss}\n")
        log_file.flush()
    avg_loss /= len(train_loader)
    avg_correct = count_correct / (count_wrong + count_correct)
    log_file.write(f"train loss: {avg_loss}\n")
    log_file.write(f"train acc: {avg_correct}\n")


def test(model, criterion, test_loader, current_epoch):
    loss = 0
    correct = 0
    wrong = 0
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, targets)
            loss += batch_loss
            predicts = torch.argmax(outputs, dim=1)
            correct += ((predicts == targets).sum())
            wrong += ((predicts != targets).sum())
    avg_loss = loss/len(test_loader)
    avg_correct = correct / (wrong+correct)
    log_file.write(f"test loss: {avg_loss}\n")
    log_file.write(f"test acc: {avg_correct}\n")
    global best_acc
    if avg_correct > best_acc:
        best_acc = avg_correct
        state = {'state_dict': model.state_dict(),
                 'current_epoch': current_epoch,
                 'acc': avg_correct}
        torch.save(state, f"checkpoint.pth.tar")


if __name__ == '__main__':
    args = args.parse_args()
    train_loader = get_train_loader(args.dataset)
    test_loader = get_test_loader(args.dataset)
    learning_rate = 1e-1
    transfer_ratio = 1e-2
    model = TVit('B_16', img_size=112, backbone_embedding=768, count_classes=10 if args.dataset == 'cifar10' else 100)
    model.to(device)
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
        log_file.write(f"Epoch {id} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        train_one_epoch(model, criterion, optimizer, get_train_loader(args.dataset))
        test(model, criterion, get_test_loader(args.dataset), id)
