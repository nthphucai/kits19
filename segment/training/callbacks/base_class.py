from typing import Union

import torch.nn as nn

from ..trainer.base_trainer import BaseTrainer


class Callback:
    def __init__(self):
        self.trainer = None
        self.models = None
        self.optimizers = None
        self.training_config = None

    def on_train_begin(self, **train_configs):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_training_batch_begin(self, epoch, step, data):
        pass

    def on_training_batch_end(self, epoch, step, data, caches=None, logs=None):
        pass

    def on_validation_batch_begin(self, epoch, step, data):
        pass

    def on_validation_batch_end(self, epoch, step, data, caches=None, logs=None):
        pass

    def set_trainer(self, trainer: BaseTrainer):
        self.models = [
            m.module if isinstance(m, nn.DataParallel) else m for m in trainer.model
        ]
        self.optimizers = [trainer.optimizer]
        self.trainer = trainer

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"

    def __str__(self):
        return repr(self)
