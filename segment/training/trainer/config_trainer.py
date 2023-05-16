import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)

from ...models import model_maps
from ...training.callbacks import callback_maps
from ...training.losses import loss_maps
from ...training.metrics import metric_maps
from ...training.optimizers import optimizer_maps
from ...training.trainer.standard_trainer import Trainer


class ConfigTrainer:
    def __init__(
        self,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset],
        model: Optional[nn.Module] = None,
        config: dict = None,
        save_config_path: str = None,
        verbose: bool = False,
        num_train_epochs: int = 2,
        out_dir: str = None,
        log_dir: str = None,
        fp16: bool = False,
        do_train: bool = True,
        do_eval: bool = False,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
    ):
        self.config = config
        self.save_config_path = save_config_path

        loss_config = config["criterion"]
        metrics_configs = config.get("metric", [])
        opt_config = config["optimizer"]
        callbacks_configs = config.get("callbacks", [])
        model_config = config.get("model", []) if model is None else None

        self.num_train_epochs = num_train_epochs
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.fp16 = fp16

        print("creating train, valid loader") if verbose else None
        dl_configs = {
            "train_batch_size": per_device_train_batch_size,
            "eval_batch_size": per_device_eval_batch_size,
        }
        self.dl_train = self.get_train_dataloader(train_dataset, **dl_configs)
        self.dl_valid = (
            self.get_eval_dataloader(valid_dataset, **dl_configs)
            if valid_dataset is not None
            else None
        )
        minibatch = next(iter(self.dl_train))
        print("data shape:", minibatch[0].shape)

        print("creating model") if verbose else None
        self.model = self._get_model(model_config) if model is None else model
        if verbose:
            print("printing model to model.txt")

            with open("model.txt", "w") as handle:
                handle.write(str(self.model))

            num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("parameters:", f"{num:,}")

        self.loss = self._get_loss(loss_config=loss_config)
        print("loss: ", self.loss) if verbose else None

        self.metrics = self._get_metrics(metrics_configs=metrics_configs)
        print("metrics: ", self.metrics) if verbose else None

        self.optimizer = self._get_optimizer(opt_config=opt_config)
        self.optimizer = optimizer_maps["look_ahead"](self.optimizer, k=5, alpha=0.5)
        print("optimizer: ", self.optimizer) if verbose else None

        self.scheduler = None

        self.callbacks = self._get_callbacks(callbacks_configs)
        print("callbacks: ", self.callbacks) if verbose else None

        self.mode = ("train", "eval") if do_eval and do_train else ("train",)

    def get_train_dataloader(self, dataset, **kwargs) -> DataLoader:
        shuffle = kwargs.get("shuffle", True)
        workers = kwargs.get("workers", True)
        pin_memory = kwargs.get("pin_memory", True)
        drop_last = kwargs.get("drop_last", True)
        batch_size = kwargs.get("train_batch_size", 2)

        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def get_eval_dataloader(self, dataset, **kwargs) -> DataLoader:
        shuffle = kwargs.get("shuffle", False)
        workers = kwargs.get("workers", True)
        pin_memory = kwargs.get("pin_memory", True)
        drop_last = kwargs.get("drop_last", True)
        batch_size = kwargs.get("eval_batch_size", 2)

        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def _get_model(self, model_config):
        name = model_config["path"]
        kwargs = self.get_kwargs(model_config, ["name"])
        model = model_maps[name](**kwargs)
        if "checkpoint" in model_config:
            w = torch.load(model_config["checkpoint"], map_location="cpu")
            print(model.load_state_dict(w, strict=False))

        if model_config.get("parallel", False):
            model = nn.DataParallel(model).cuda()
        elif model_config.get("gpu", False):
            model = model.cuda()

        return model

    def _get_loss(self, loss_config):
        name = loss_config["name"]
        kwargs = self.get_kwargs(loss_config, ["name", "class_weight"])
        class_weight = np.load(loss_config["class_weight"])
        print("class_weight:", np.unique(class_weight))
        loss = loss_maps[name](class_weight=class_weight, **kwargs)
        return loss

    def _get_metrics(self, metrics_configs):
        name = metrics_configs["name"]
        kwargs = self.get_kwargs(metrics_configs, ["name"])
        metrics = metric_maps[name](**kwargs)
        return metrics

    def _get_optimizer(self, opt_config):
        name = opt_config["name"]
        opt = optimizer_maps[name]
        kwargs = self.get_kwargs(opt_config, ["name"])

        if "checkpoint" in opt_config:
            opt.load_state_dict(
                torch.load(opt_config["checkpoint"], map_location="cpu")
            )

        opt = opt([p for p in self.model.parameters() if p.requires_grad], **kwargs)
        return opt

    def _get_callbacks(self, callbacks_configs):
        callbacks_configs = [] + callbacks_configs
        cbs = []

        for cb_config in callbacks_configs:
            name = cb_config["name"].lower()
            kwargs = self.get_kwargs(cb_config)
            if name in callback_maps:
                cbs.append(callback_maps[name](**kwargs))

        return cbs

    def train(self):
        """
        run the training process based on config
        """
        runner = Trainer(
            train_data=self.dl_train,
            val_data=self.dl_valid,
            model=self.model,
            loss=self.loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=self.metrics,
            num_train_epochs=self.num_train_epochs,
            out_dir=self.out_dir,
            log_dir=self.log_dir,
            fp16=self.fp16,
        )
        if self.save_config_path is not None:
            full_file = f"{self.save_config_path}"
            with open(full_file, "w") as handle:
                json.dump(self.config, handle)

        return runner.run(mode=self.mode, callbacks=self.callbacks)

    @staticmethod
    def get_kwargs(configs, excludes=("name",)):
        excludes = set(excludes)

        return {k: configs[k] for k in configs if k not in excludes}


def create(config, save_config_path=None, name=None, verbose=1):
    """
    create runner
    Args:
        config: config dict or path to config dict
        save_config_path: where to save config after training
        name: name of saved config
        verbose: print creation step

    Returns: ConfigRunner
    """
    if not isinstance(config, dict):
        with open(config) as handle:
            config = json.load(handle)

    return ConfigTrainer(
        config, save_config_path=save_config_path, name=name, verbose=verbose
    )
