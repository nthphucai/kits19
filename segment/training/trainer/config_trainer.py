import json

import numpy as np
import torch
import torch.nn as nn

from ...training.callbacks import callback_maps
from ...training.losses import loss_maps
from ...training.metrics import metric_maps
from ...training.optimizers import optimizer_maps
from ...training.schedulers import scheduler_maps
from ...training.trainer.standard_trainer import Trainer
from ...utils import parameter as para


class ConfigTrainer:
    def __init__(self, data, model, config, save_config_path=None, verbose=1):
        self.config = config
        self.save_config_path = save_config_path

        loss_config = config["criterion"]
        metrics_configs = config.get("metric", [])
        opt_config = config["optimizer"]
        callbacks_configs = config.get("callbacks", [])
        scheduler_configs = config.get("schedulers", [])
        # model_config = config.get("model", [])

        print("creating train, valid loader") if verbose else None
        self.dl_train, self.dl_valid = self._get_repo(data)
        if verbose:
            print("train: ", len(self.dl_train))
            print("valid: ", len(self.dl_valid)) if self.dl_valid is not None else None

        print("creating model") if verbose else None
        self.model = model  # self._get_model(model, model_config)
        if verbose:
            print("printing model to model.txt")

            with open("model.txt", "w") as handle:
                handle.write(str(self.model))

            num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("parameters:", f"{num:,}")

        class_weight = np.array([0.00673965, 0.27936378, 1.0])
        # class_weight = np.array([0.27936378, 1.0])
        class_weight = class_weight.reshape(1, para.num_classes, *[1] * 3)
        self.loss = self._get_loss(
            loss_config=loss_config,
            smooth=1e-6,
            label_smoothing=None,
            class_weight=class_weight,
            device="cpu",
        )
        print("loss: ", self.loss) if verbose else None

        self.metrics = self._get_metrics(metrics_configs=metrics_configs, binary=False)
        print("metrics: ", self.metrics) if verbose else None

        self.optimizer = self._get_optimizer(
            opt_config=opt_config, lr=para.learning_rate
        )
        print("optimizer: ", self.optimizer) if verbose else None

        self.callbacks = self._get_callbacks(callbacks_configs)
        print("callbacks: ", self.callbacks) if verbose else None

    def _get_repo(self, loaders):
        if isinstance(loaders, (tuple, list)):
            train, valid = loaders
        else:
            train = loaders
            valid = None

        return train, valid

    def _get_model(self, model, model_config):
        if "checkpoint" in model_config:
            w = torch.load(model_config["checkpoint"], map_location="cpu")
            print(model.load_state_dict(w, strict=False))

        if model_config.get("parallel", False):
            model = nn.DataParallel(model).cuda()
        elif model_config.get("gpu", False):
            model = model.cuda()

        return model

    def _get_loss(self, loss_config, **kwargs):
        loss = loss_maps[loss_config](**kwargs)
        return loss

    def _get_metrics(self, metrics_configs, **kwargs):
        metrics = metric_maps[metrics_configs](**kwargs)
        return metrics

    def _get_optimizer(self, opt_config, **kwargs):
        opt = optimizer_maps[opt_config]
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
            scheduler=None,
            metric=self.metrics,
            callbacks=None,
            num_train_epochs=5,
            output_dir=None,
        )
        if self.save_config_path is not None:
            full_file = f"{self.save_config_path}"
            with open(full_file, "w") as handle:
                json.dump(self.config, handle)

        return runner.run()


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
