import torch.utils.data
import torchvision.datasets as datasets
from torchvision import transforms

batch_size = 128
img_size = 224
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_train_set(name):
    """获取一个数据集的dataset"""
    if name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    else:
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    return train_set


def get_train_loader(name):
    dataset = get_train_set(name)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def get_test_set(name):
    """获取一个数据集的测试集"""
    if name == "cifar10":
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    else:
        test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    return test_set


def get_test_loader(name):
    dataset = get_test_set(name)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    return loader
