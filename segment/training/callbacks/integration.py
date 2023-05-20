import wandb

from .base_class import TrainerCallback

class WandbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """

    def __init__(self):
        self._wandb = wandb
        # log outputs
        # self._log_model = os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def setup(self, args=None, state=None, model=None, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training. Use along with
                `TrainingArguments.load_best_model_at_end` to upload best model.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        pass

    def on_train_begin(self, args=None, state=None, control=None, model=None, **kwargs):
        self.setup(args=None, state=None, model=None, **kwargs)

    def on_train_end(self, args=None, state=None, control=None, model=None, **kwargs):
        pass

    def on_epoch_end(self, e: int, logs=None):
        self.on_log(args=None, state=None, control=None, model=None, logs=logs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        self.setup(args, state, model)
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
