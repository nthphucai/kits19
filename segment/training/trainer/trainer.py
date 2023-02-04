import time
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from segment.data.augs import augs
from segment.data.data_loaders import get_dloader
from segment.training.callbacks.utils import Callbacks
from segment.utils.utils import get_progress

# from segment.training.models.optimizer import Lookahead


class Trainer:
    def __init__(
        self,
        repo,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: nn.Module,
        scheduler: nn.Module,
        metric: nn.Module,
        fold: Optional[int] = None,
        num_epochs: int = 4,
        verbose: bool = False,
        save_path: dict = None,
    ):
        # configs: dict) :

        self.repo = repo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.opt = optimizer
        # self.base_opt = optimizer
        # self.opt = Lookahead(self.base_opt, k=5, alpha=0.5)
        self.scheduler = scheduler
        self.metric = metric
        self.fold = fold
        print("fold", self.fold)
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.path = save_path
        print("verbose:", self.verbose)
        # _iter: no.of.batch
        self._iter = 2
        self.best_loss = float("inf")

        self.dl_train, self.dl_val = get_dloader(df=self.repo, fold=self.fold)
        self.callbacks = Callbacks(self.fold)
        # if configs is not None: self.callbacks.save_config(configs)

        self.since = time.time()

    def _loss_and_outputs(self, imgs: torch.Tensor, targets: torch.Tensor):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        preds = self.model(imgs)
        loss = self.criterion(preds, targets)

        return loss, preds

    def minibatch(self):
        """check repo + model"""
        self.model.train()
        imgs, targets = next(iter(self.dl_train))
        for epoch in range(self.num_epochs):
            print("epoch:", epoch)
            loss, preds = self._loss_and_outputs(imgs, targets)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            print("loss:", loss.item())

    def _do_train_epoch(self, epoch: int):
        # scaler = GradScaler()
        self.model.train()
        self.metric.init_list()
        running_loss = 0.0

        for itr, (imgs, trues) in get_progress(
            enumerate(self.dl_train), total=len(self.dl_train)
        ):
            # # Using Mix-precision training
            # with autocast():
            #   loss, preds = self._loss_and_outputs(imgs, trues)
            # scaler.scale(loss).backward()

            loss, preds = self._loss_and_outputs(imgs, trues)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # if (itr+1) % self._iter == 0:
            #   scaler.step(self.opt)
            #   scaler.update()
            #   self.opt.zero_grad()

            running_loss += loss.item()

            self.metric.update(
                preds.detach().cpu().numpy(), trues.detach().cpu().numpy()
            )

        train_loss = running_loss / len(self.dl_train)
        dsc_batch = self.metric.get_dsc_batch()
        dsc_organ, dsc_tumor = self.metric.get_dsc_class()

        return train_loss, dsc_batch, dsc_organ, dsc_tumor

    def _do_val_epoch(self, epoch: int):
        self.model.eval()
        self.metric.init_list()
        running_loss = 0.0

        with torch.no_grad():
            for itr, (imgs, trues) in get_progress(
                enumerate(self.dl_val), total=len(self.dl_val)
            ):
                loss, preds = self._loss_and_outputs(imgs, trues)
                running_loss += loss.item()
                self.metric.update(
                    preds.detach().cpu().numpy(), trues.detach().cpu().numpy()
                )

        val_loss = running_loss / len(self.dl_val)
        dsc_batch = self.metric.get_dsc_batch()
        dsc_organ, dsc_tumor = self.metric.get_dsc_class()

        return val_loss, dsc_batch, dsc_organ, dsc_tumor

    def run(self, mode=["train", "val"]):
        for epoch in range(self.num_epochs):
            for phase in mode:
                if phase == "train":
                    loss, dsc_batch, dsc_organ, dsc_tumor = self._do_train_epoch(epoch)
                    if self.scheduler:
                        self.scheduler.step(loss) if "val" not in mode else None
                    logs = {
                        "train_loss": loss,
                        "train_dsc_organ": dsc_organ,
                        "train_dsc_tumor": dsc_tumor,
                    }

                elif phase == "val":
                    loss, dsc_batch, dsc_organ, dsc_tumor = self._do_val_epoch(epoch)
                    if self.scheduler:
                        self.scheduler.step(loss)
                    _logs = {
                        "val_loss": loss,
                        "val_dsc_organ": dsc_organ,
                        "val_dsc_tumor": dsc_tumor,
                    }
                    logs.update(_logs)

                result = {
                    "Loss": f"{loss:.4f}",
                    "DSC": f"{dsc_batch: 4f}",
                    "DSC_organ": f"{dsc_organ:.4f}",
                    "DSC_tumor": f"{dsc_tumor:.4f}",
                }

                """ Display values
          """
                print("Epoch", f"{epoch}/" + f"{self.num_epochs}", end=" ")
                for key, value in zip(result.keys(), result.values()):
                    print(key, ":", value, ",", end=" ")
                print("\n")
            """
        ""save model
        """
            if self.verbose:
                csv_file, filename = self.callbacks.save_logs(
                    epoch, logs=logs, name=self.path["save_log_path"]
                )
                if epoch % 1 == 0:
                    save_loss = float(result.get("Loss"))
                    if (save_loss < self.best_loss) and (epoch > 2):
                        self.best_loss = save_loss
                        self.callbacks.save_model(
                            self.best_loss,
                            epoch,
                            self.model,
                            self.opt,
                            name=self.path["save_model_path"],
                        )

        duration = (time.time() - self.since) / 60
        print("running_time:", f"{duration:.4f}")
        """
      plot results
      """
        if self.verbose:
            self.callbacks.plots(results=filename)
