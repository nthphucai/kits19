from ..trainer.base_trainer import BaseTrainer


class TrainerCallback:
    def __init__(self):
        self.trainer = None
        self.models = None
        self.optimizers = None
        self.training_config = None

    def on_train_begin(self, **train_configs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, epoch):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_training_batch_begin(self, epoch, step, data):
        """
        Event called at the beginning of a training batch.
        """
        pass

    def on_training_batch_end(self, epoch, step, data, caches=None, logs=None):
        """
        Event called at the end of a training batch.
        """
        pass

    def on_validation_batch_begin(self, epoch, step, data):
        """
        Event called at the beginning of a validation batch.
        """
        pass

    def on_validation_batch_end(self, epoch, step, data, caches=None, logs=None):
        """
        Event called at the end of a validation batch.
        """
        pass

    def on_save(self):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, logs):
        """
        Event called after logging the last logs.
        """
        pass

    def set_trainer(self, trainer: BaseTrainer):
        # self.models = [
        #     m.module if isinstance(m, nn.DataParallel) else m for m in trainer.model
        # ]
        self.models = [trainer.model]
        self.optimizers = [trainer.optimizer]
        self.trainer = trainer

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"

    def __str__(self):
        return repr(self)
