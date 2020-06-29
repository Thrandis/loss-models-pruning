import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)
from torchvision.datasets import CIFAR10, MNIST

from pruning import mask_net


DATA_PATH = os.path.join(os.environ['SLURM_TMPDIR'], 'data')


def train_loop(net, optimizer, loader, mask=None):
    """Loop for network training (classification)."""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if mask is not None:
            mask_net(net, mask)
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # Monitoring
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_norm = train_loss / (batch_idx + 1)
        ratio = float(total - correct) / total
    optimizer.zero_grad()
    if mask is not None:
        mask_net(net, mask)
    return train_loss_norm, ratio


@torch.no_grad()
def eval_loop(net, loader, train=False):
    """Loop for network evalutaion (classification)."""
    if train:
        net.train()
    else:
        net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = net(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Monitoring
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_norm = test_loss / (batch_idx + 1)
        ratio = float(total - correct) / total
    return test_loss_norm, ratio


def kaiming_init(net, tanh=False):
    """Initialize network using Kaiming He initialization."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if tanh:
                torch.nn.init.xavier_normal_(m.weight)
            else:
                torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.affine:
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)


def get_mnist_sets(seed, data_path=DATA_PATH):
    """Returns train, valid and test splits for MNIST."""
    train = MNIST(root=data_path, train=True, download=True,
                  transform=ToTensor())
    test = MNIST(root=data_path, train=False, download=True,
                 transform=ToTensor())
    # Fork the rng such that the splits depend on the seed, not on the state.
    with torch.random.fork_rng(devices=['cuda']):
        torch.random.manual_seed(seed)
        train, valid = random_split(train, [50000, 10000])
    # Get features and targets
    train_features, train_targets = next(iter(DataLoader(train, len(train))))
    valid_features, valid_targets = next(iter(DataLoader(valid, len(valid))))
    test_features, test_targets = next(iter(DataLoader(test, len(test))))
    # Apply normalisation
    m = train_features.mean()
    s = train_features.std()
    train_features = (train_features.view(len(train), -1) - m) / s
    valid_features = (valid_features.view(len(valid), -1) - m) / s
    test_features = (test_features.view(len(test), -1) - m) / s
    # Move to GPU and pack into TensorDataset
    train = TensorDataset(train_features.to('cuda'), train_targets.to('cuda'))
    valid = TensorDataset(valid_features.to('cuda'), valid_targets.to('cuda'))
    test = TensorDataset(test_features.to('cuda'), test_targets.to('cuda'))
    return train, valid, test


def get_mnist_loader(which_set, bs=100, num_workers=0, shuffle=False,
                     data_path=DATA_PATH, which_transform=None):
    """Returns a dataloader of MNIST"""
    if which_transform is None:
        which_transform = which_set
    if which_transform in ['train', 'valid', 'test']:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    else:
        raise NotImplementedError
    if which_set == 'train':
        dataset = Subset(MNIST(root=data_path, train=True, download=True,
                               transform=transform),
                         range(55000))
    elif which_set == 'valid':
        dataset = Subset(MNIST(root=data_path, train=True, download=True,
                               transform=transform),
                         range(55000, 60000))
    elif which_set == 'test':
        dataset = MNIST(root=data_path, train=False, download=True,
                        transform=transform)
    else:
        raise NotImplementedError
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader


def get_mnist_loader_gpu(which_set, bs=100, num_workers=0, shuffle=False,
                         data_path=DATA_PATH, which_transform=None):
    """Returns a dataloader of MNIST"""
    loader = get_mnist_loader(which_set, bs, 0, False, data_path,
                              which_transform)
    features = []
    targets = []
    for i, (f, t) in enumerate(loader):
        features.append(f)
        targets.append(t)
    features = torch.cat(features, dim=0).to('cuda')
    targets = torch.cat(targets, dim=0).to('cuda')
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                        num_workers=0)
    return loader


def get_cifar10_loader(which_set, bs=100, num_workers=0, shuffle=False,
                       data_path=DATA_PATH, which_transform=None):
    """Returns a dataloader of CIFAR10."""
    if which_transform is None:
        which_transform = which_set
    if which_transform == 'train':
        transform = Compose([RandomCrop(32, padding=4),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             Normalize((0.4914, 0.4822, 0.4465),
                                       (0.247, 0.243, 0.261))])
    elif which_transform in ['valid', 'test']:
        transform = Compose([ToTensor(),
                             Normalize((0.4914, 0.4822, 0.4465),
                                       (0.247, 0.243, 0.261))])
    else:
        raise NotImplementedError
    if which_set == 'train':
        dataset = Subset(CIFAR10(root=data_path, train=True, download=True,
                                 transform=transform),
                         range(45000))
    elif which_set == 'valid':
        dataset = Subset(CIFAR10(root=data_path, train=True, download=True,
                                 transform=transform),
                         range(45000, 50000))
    elif which_set == 'test':
        dataset = CIFAR10(root=data_path, train=False, download=True,
                          transform=transform)
    else:
        raise NotImplementedError
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader


def get_cifar10_loader_gpu(which_set, bs=100, num_workers=0, shuffle=False,
                           data_path=DATA_PATH, which_transform=None):
    """Returns a dataloader of MNIST"""
    loader = get_cifar10_loader(which_set, bs, 0, False, data_path,
                                'test')
    features = []
    targets = []
    for i, (f, t) in enumerate(loader):
        features.append(f)
        targets.append(t)
    features = torch.cat(features, dim=0).to('cuda')
    targets = torch.cat(targets, dim=0).to('cuda')
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                        num_workers=0)
    return loader
