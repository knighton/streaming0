import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.functional import to_tensor
from typing import Any, Callable, Dict, Optional, Tuple

from .index import get_index_basename, MDSIndex


class LocalMDSDataset(Dataset):
    def __init__(self, dirname: str) -> None:
        self.dirname = dirname

        filename = os.path.join(dirname, get_index_basename())
        self.index = MDSIndex.load(open(filename))

        samples_per_shard = np.zeros(len(self.index.shards), np.int64)
        for i, shard in enumerate(self.index.shards):
            samples_per_shard[i] = shard.samples
        self.shard_offsets = np.concatenate([np.zeros(1, np.int64), samples_per_shard.cumsum()])

    def __len__(self):
        return self.shard_offsets[-1]

    def _find_shard(self, idx: int) -> Tuple[int, int]:
        low = 0
        high = len(self.shard_offsets) - 1
        while True:
            if low + 1 == high:
                if idx == self.shard_offsets[high]:
                    shard = high
                else:
                    shard = low
                break
            mid = (low + high) // 2
            div = self.shard_offsets[mid]
            if idx < div:
                high = mid
            elif div < idx:
                low = mid
            else:
                shard = mid
                break
        offset = idx - self.shard_offsets[shard]
        return shard, offset

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index."""
        shard, idx_in_shard = self._find_shard(idx)
        filename = os.path.join(self.dirname, self.index.shards[shard].raw.basename)
        offset = (1 + idx_in_shard) * 4
        with open(filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        return self.index.decode_sample(data)


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, x: Any, y: Any) -> Tuple[Any, Any]:
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class LocalMDSVisionDataset(LocalMDSDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        dirname = os.path.join(root, split)
        super().__init__(dirname)

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError('Only transforms or transform/target_transform can be passed as ' +
                             'argument')

        self.transform = transform
        self.target_transform = target_transform

        if not has_transforms:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x, y)


class LocalCIFAR10(LocalMDSVisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        transform = transform or to_tensor
        super().__init__(root, split, None, transform, target_transform)
