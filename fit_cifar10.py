from argparse import ArgumentParser

from streaming.base.mds.dataset import LocalCIFAR10


def parse_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--split', type=str, default='train')
    return args.parse_args()


def main(args):
    dataset = LocalCIFAR10(args.dataset, args.split)
    for i in range(len(dataset)):
        x, y = dataset[i]
        print('x =', x.shape, x)
        print('y =', y)
        print()


if __name__ == '__main__':
    main(parse_args())
