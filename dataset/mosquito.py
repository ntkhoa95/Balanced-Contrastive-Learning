from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image
from typing import Optional, Callable, Tuple, Any

class MosquitoDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        phase: str = "train"
    ):
        super().__init__(
            root,
            transform
        )
        assert phase in ["train", "val"]
        self.phase = phase
        self.labels = self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if self.phase == "train":
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                sample3 = self.transform[2](sample)
                sample = [sample1, sample2, sample3]
            else:
                sample = self.transform(sample)
        return sample, target
