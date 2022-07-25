from argparse import ArgumentParser

from streaming.vision.cifar10 import CIFAR10


def parse_args():
    args = ArgumentParser()
    args.add_argument('--remote', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--local', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--split', type=str, default='train')
    args.add_argument('--shuffle', type=int, default=1)
    return args.parse_args()


def main(args):
    dataset = CIFAR10(args.local, args.remote, args.split, args.shuffle)
    for x, y in dataset:
        print('x =', x.shape, x)
        print('y =', y)
        print()


if __name__ == '__main__':
    main(parse_args())
