#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '進捗バーのテスト'
#

import time


def main():

    bar = '#' * 10
    no_bar = ' ' * 10

    print(bar)
    print(no_bar)

    max_num = 100
    for i in range(max_num):
        time.sleep(0.05)
        print(
            f'\r [{bar[:i*10//max_num]}{no_bar[i*10//max_num:]}] {i*100//max_num:3}%',
            end='', flush=True
        )

    print('\rfinish            ')


if __name__ == '__main__':
    main()
