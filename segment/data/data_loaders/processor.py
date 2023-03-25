import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)

from ..data_readers.data_reader import DatasetReader


class DataProcessor:
    def __init__(self, 
        batch_size:int=2,
        workers:int=2,
        pin_memory:bool=True,
        drop_last=True,
        shuffle=True,
        sampler=True,
    ):
      self.batch_size = batch_size
      self.workers = workers
      self.pin_memory = pin_memory
      self.drop_last = drop_last
      self.shuffle = shuffle
      self.sampler = sampler
        
    def get_dloader(self, dataset):
      sampler = (self.sampler or RandomSampler(self) if self.shuffle else SequentialSampler(self))
      return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
      )

    


      