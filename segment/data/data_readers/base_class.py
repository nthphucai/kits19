from abc import abstractmethod

from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)

import segment.utils.parameter as para


class StandardDataset(Dataset):
    def __init__(self):
        super(StandardDataset, self).__init__()
        self.sampler = None

    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        return self.get_item(idx)

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    def get_loader(
        self,
        batch_size=para.bz,
        workers=para.workers,
        pin=True,
        drop_last=True,
        shuffle=True,
    ):
        sampler = (
            self.sampler or RandomSampler(self) if shuffle else SequentialSampler(self)
        )
        return DataLoader(
            self,
            batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin,
            drop_last=drop_last,
        )
