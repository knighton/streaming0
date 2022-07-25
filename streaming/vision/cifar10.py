from typing import Callable, List, Optional

from .base import MDSVisionDataset


class CIFAR10(MDSVisionDataset):
    def __init__(
        self,
        local: str,
        remote: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        prefetch: Optional[int] = 100_000,
        keep_zip: Optional[bool] = True,
        retry: int = 2,
        timeout: float = 60,
        shard_hashes: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> None:
        super().__init__(local, remote, split, shuffle, None, transform, target_transform, prefetch,
                         keep_zip, retry, timeout, shard_hashes, batch_size)