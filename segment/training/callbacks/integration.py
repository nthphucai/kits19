import wandb

from .base_class import TrainerCallback

class WandbCallback(TrainerCallback):
    def __init__(self):
        self._wandb = wandb

    def setup(self, args=None, state=None, model=None, **kwargs):
        pass

    def on_train_begin(self, **train_configs):
        pass

    def on_training_batch_end(self, epoch, step, data, caches=None, logs=None):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, e: int, logs=None):
        self.on_log(logs=logs)

    def on_log(self, logs=None):
        self.setup()
        # logs = self.rewrite_logs(logs)
        self._wandb.log({**logs})

    @staticmethod
    def rewrite_logs(d):
        new_d = {}
        eval_prefix = "eval_"
        eval_prefix_len = len(eval_prefix)
        for k, v in d.items():
            if k.startswith(eval_prefix):
                new_d["eval/" + k[eval_prefix_len:]] = v
            else:
                new_d["train/" + k] = v
        return new_d
