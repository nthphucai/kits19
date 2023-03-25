from abc import abstractmethod

from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)


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
        