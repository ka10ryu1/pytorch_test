#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'メイン学習部分'
#
# https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py

import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Lib.net import Net
from Lib.func import argsPrint, getFilePath


def bar(i, num):
    bar = '#' * 10
    no_bar = ' ' * 10
    print(
        f'\r [{bar[:i*10//num]}{no_bar[i*10//num:]}] {i*100//num:3}%',
        end='', flush=True
    )


def getTime(st):
    timer = time.time() - st
    ext = 's'
    if timer > 60:
        timer /= 60
        ext = 'min'

    if timer > 60 and ext == 'min':
        timer /= 60
        ext = 'h'

    if timer > 24 and ext == 'h':
        timer /= 24
        ext = 'day'

    return f'{timer:.1f}[{ext}]'


class trainer(object):

    def __init__(self, args, model, device, train_loader, test_loader, optimizer):
        self.log_interval = args.log_interval
        self.model = model
        self.device = device
        self.loader = {'train': train_loader, 'test': test_loader}
        self.optimizer = optimizer
        self.epochs = args.epochs

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self._train(epoch)
            self._test()

    def save(self, save_dir):
        path = getFilePath(save_dir, 'mnist', '.model')
        torch.save(self.model.state_dict(), path)

    def _train(self, epoch):
        self.model.train()
        loader = self.loader['train']
        st = time.time()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                bar(batch_idx, len(loader))

        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({})'.format(
            epoch, batch_idx * len(data), len(loader.dataset),
            100. * batch_idx / len(loader), loss.item(), getTime(st))
        )

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        loader = self.loader['test']
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset))
        )


def command():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    argsPrint(args)
    return args


def main():
    args = command()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose(
                           [transforms.ToTensor(), transforms.Normalize(
                               (0.1307,), (0.3081,))]
                       )),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose(
                           [transforms.ToTensor(), transforms.Normalize(
                               (0.1307,), (0.3081,))]
                       )),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )

    mnist = trainer(args, model, device, train_loader, test_loader, optimizer)
    mnist.run()
    mnist.save('./result')


if __name__ == '__main__':
    main()
