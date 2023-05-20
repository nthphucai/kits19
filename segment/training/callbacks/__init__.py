from .clr import LrFinder, SuperConvergence, WarmRestart
from .integration import WandbCallback
from .standard_callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                                 ReduceLROnPlateau, Tensorboard)

callback_maps = {
    "csv_logger": CSVLogger,
    "tensorboard": Tensorboard,
    "plateau": ReduceLROnPlateau,
    "early_stopping": EarlyStopping,
    "checkpoint": ModelCheckpoint,
    "super_convergence": SuperConvergence,
    "lr_finder": LrFinder,
    "warm_restart": WarmRestart,
    "report_to_wandb": WandbCallback
}
