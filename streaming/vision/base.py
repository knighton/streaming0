import os
from torchvision.transforms.functional import to_tensor
from typing import Any, Callable, Optional, Tuple

from ..base.mds.dataset import MDSDataset


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, x: Any, y: Any) -> Tuple[Any, Any]:
        if self.transform:
            x = self.transform(x)
        else:
            x = to_tensor(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class MDSVisionDataset(MDSDataset):
    def __init__(
        self,
        remote: Optional[str],
        local: str,
        shuffle: bool,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        prefetch: Optional[int] = 100_000,
        keep_zip: bool = True,
        retry: int = 2,
        timeout: float = 60,
        batch_size: Optional[int] = None
    ) -> None:
        super().__init__(remote, local, shuffle, prefetch, keep_zip, retry, timeout, batch_size)

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




