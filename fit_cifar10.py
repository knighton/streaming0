from argparse import ArgumentParser

from streaming.vision.cifar10 import CIFAR10


def parse_args():
    args = ArgumentParser()
    args.add_argument('--local', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--remote', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--split', type=str, default='train')
    args.add_argument('--shuffle', type=int, default=1)
    args.add_argument('--shard_hashes', type=str, default='xxh64')
    return args.parse_args()


def main(args):
    shard_hashes = args.shard_hashes.split(',') if args.shard_hashes else []
    dataset = CIFAR10(args.local, args.remote, args.split, args.shuffle, shard_hashes=shard_hashes)
    for x, y in dataset:
        print('x =', x.shape, x)
        print('y =', y)
        print()


if __name__ == '__main__':
    main(parse_args())
