import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from ...utils.file_utils import logging
from ..callbacks.utils import plot_df, save_logs, save_model
from .base_trainer import BaseTrainer
from .utils import get_dict


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_data: Iterable,
        val_data: Iterable,
        model: nn.Module,
        loss: nn.Module,
        optimizer: nn.Module,
        scheduler: nn.Module,
        metric: nn.Module,
        callbacks: Optional[List[nn.Module]],
        num_train_epochs: int,
        output_dir: str,
        save_model: bool = False,
        fp16: bool = False,
        fold: Optional[int] = None,
    ):

        super().__init__(
            model, train_data, val_data, loss, optimizer, scheduler, metric
        )

        self.dl_train = train_data
        self.dl_val = val_data
        self.loss = loss
        self.opt = optimizer
        self.scheduler = scheduler
        self.score = metric
        self.callbacks = callbacks

        self.save_model = save_model
        self.output_dir = output_dir

        self.best_loss = float("inf")
        self.num_train_epochs = num_train_epochs

        self.scaler = torch.cuda.amp.GradScaler()
        self.gradient_accumulation = 1

        dates = (datetime.datetime.now()).strftime("%Y%m%d")
        remaining_name = f"fold_{fold}_{dates}"
        self.log_path = Path(f"{output_dir}", "logs", remaining_name + ".csv")
        self.model_path = Path(
            f"{self.output_dir}", "checkpoints", remaining_name + ".pt"
        )

    def train_mini_batch(self):
        self.model.train()
        imgs, targets = next(iter(self.dl_train))
        for iter in range(self.num_train_epochs):
            loss, _ = self._train_one_batch(iter, imgs, targets)
            print("loss:", loss.item())

    def _measures_one_batch(self):
        dsc_batch = self.score.get_dsc_batch()
        dsc_organ, dsc_tumor = self.score.get_dsc_class()
        return dsc_batch, dsc_organ, dsc_tumor

    def _train_one_batch(self, iter, imgs, targets):
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                loss, preds = self.loss_and_output(imgs, targets)
                self.scaler.scale(loss).backward()
                if (iter+1) % self.gradient_accumulation == 0: 
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
        else:
            loss, preds = self.loss_and_output(imgs, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss, preds

    def _eval_one_batch(self, imgs, targets):
        loss, preds = self.loss_and_output(imgs, targets)
        return loss, preds

    def run(self, mode=["train", "valid"], callbacks=None):
        callbacks = callbacks or []

        [c.set_trainer(self) for c in callbacks]
        train_configs = {
            "train_loader": self.dl_train,
            "test_loader": self.dl_val,
            "start_epoch": 1,
        }
        [c.on_train_begin(**train_configs) for c in callbacks]

        for e in range(self.num_train_epochs):
            print("\nepoch", f"{e}/{self.num_train_epochs}")
            for _ in mode:
                [c.on_epoch_begin(e) for c in callbacks]
                loss, dsc_batch, dsc_organ, dsc_tumor = self.train_one_epoch(
                    e, callbacks=callbacks
                )
                if self.scheduler:
                    self.scheduler.step(loss) if "val" not in mode else None

                logs = get_dict(
                    names=["loss", "dsc", "dsc_organ", "dsc_tumor"],
                    values=[loss, dsc_batch, dsc_organ, dsc_tumor],
                    verbose=True,
                )

                if "eval" in mode:
                    loss, dsc_batch, dsc_organ, dsc_tumor = self.val_one_epoch(e)
                    self.scheduler.step(loss) if self.scheduler else None
                    logs_ = get_dict(
                        names=["loss", "dsc_batch", "dsc_organ", "dsc_tumor"],
                        values=[loss, dsc_batch, dsc_organ, dsc_tumor],
                    )
                    logs.update(logs_)

                if self.output_dir is not None:
                    _, filename = save_logs(e, logs, self.log_path)
                    plot_df(results=filename)

                    if self.save_model and (e > 1) and (loss < self.best_loss):
                        self.best_loss = loss
                        save_model(loss, e, self.model, self.opt, self.model_path)

            [c.on_epoch_end(e, logs) for c in callbacks]

        logging.info("Save logs at directory: %s", self.log_path)
        logging.info("Save model at directory: %s", self.model_path)

        [c.on_train_end() for c in callbacks]
