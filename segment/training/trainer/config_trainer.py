import json
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as load_batch

from ...models import model_maps
from ...training.callbacks import callback_maps
from ...training.losses import loss_maps
from ...training.metrics import metric_maps
from ...training.optimizers import optimizer_maps
from ...training.schedulers import scheduler_maps
from ...training.trainer.standard_trainer import Trainer
from ...utils import parameter as para


class ConfigTrainer:
    def __init__(
        self,
        data_loaders: Union[tuple, list],
        model: Optional[nn.Module] = None,
        config: dict = None,
        save_config_path: str = None,
        verbose: bool = False,
        num_train_epochs:int=2,
        output_dir:Optional[str]=None,
        log_dir:str=None, 
        fp16:bool=False,
        fold: Optional[int]=1
    ):
        self.config = config
        self.save_config_path = save_config_path

        loss_config = config["criterion"]
        metrics_configs = config.get("metric", [])
        opt_config = config["optimizer"]
        callbacks_configs = config.get("callbacks", [])
        scheduler_configs = config.get("schedulers", [])
        model_config = config.get("model", []) if model is None else None

        self.num_train_epochs=num_train_epochs
        self.output_dir=output_dir
        self.log_dir=log_dir
        self.fp16=fp16,
        self.fold=fold

        print("creating train, valid loader") if verbose else None
        self.dl_train, self.dl_valid = self._get_dataloader(data_loaders)
        if verbose:
            print("train: ", len(self.dl_train))
            print("valid: ", len(self.dl_valid)) if self.dl_valid is not None else None

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
        print("optimizer: ", self.optimizer) if verbose else None

        self.scheduler = None
        print("optimizer: ", self.scheduler) if verbose else None

        # self.callbacks = [callback_maps["early_stopping"]()] #self._get_callbacks(callbacks_configs)
        self.callbacks = self._get_callbacks(callbacks_configs)
        print("callbacks: ", self.callbacks) if verbose else None

    def _get_dataloader(self, loaders):
        if isinstance(loaders, (tuple, list)):
            train, valid = loaders
        else:
            train = loaders
            valid = None

        return train, valid

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
        print("class_weight:", class_weight)
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
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            fp16=self.fp16,
            fold=self.fold,
        )
        if self.save_config_path is not None:
            full_file = f"{self.save_config_path}"
            with open(full_file, "w") as handle:
                json.dump(self.config, handle)

        return runner.run(mode=["train", "valid"], callbacks=self.callbacks)

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
