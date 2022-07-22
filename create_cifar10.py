import os
from argparse import ArgumentParser, Namespace
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from streaming.base.mds.writer import MDSCreator


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in', type=str, default='/datasets/cifar10/')
    args.add_argument('--out', type=str, default='/datasets/mds/cifar10/')
    args.add_argument('--splits', type=str, default='train,val')
    args.add_argument('--compression', type=str, default='')  # zstd:7')
    args.add_argument('--hashes', type=str, default='sha1,xxh64')
    args.add_argument('--limit', type=int, default=1 << 21)
    args.add_argument('--progbar', type=int, default=1)
    args.add_argument('--leave', type=int, default=0)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Main: create streaming CIFAR10 dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = args.splits.split(',')
    fields = {
        'i': 'int',
        'x': 'pil',
        'y': 'int'
    }
    hashes = args.hashes.split(',') if args.hashes else []
    for split in splits:
        dataset = CIFAR10(root=getattr(args, 'in'), train=(split == 'train'), download=True)
        indices = np.random.permutation(len(dataset))
        if args.progbar:
            indices = tqdm(indices, leave=args.leave)
        split_dir = os.path.join(args.out, split)
        with MDSCreator(split_dir, fields, args.compression, hashes, args.limit) as out:
            for i in indices:
                x = dataset.data[i]
                x = Image.fromarray(x)
                y = dataset.targets[i]
                out.write({
                    'i': i,
                    'x': x,
                    'y': y,
                })


if __name__ == '__main__':
    main(parse_args())
