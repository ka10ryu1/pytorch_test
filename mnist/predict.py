#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'メイン学習部分'
#
# https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py

import argparse
import torch
from torchvision import datasets, transforms
import numpy as np

from Lib.net import Net
from Lib.func import argsPrint, getFilePath


def main():
    torch.manual_seed(1)
    test_batch_size = 1000
    no_cuda = True
    use_cuda = not no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]
                       )),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    model = Net()
    model.load_state_dict(torch.load('./result/mnist.model'))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for (images, labels) in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    total = 0
    for i in range(len(classes)):
        print(
            f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:5.1f}%'
        )
        total += class_correct[i] / class_total[i]

    total /= len(classes)
    print(f'Total accuracy: {total*100:5.1f}%')


if __name__ == '__main__':
    main()
