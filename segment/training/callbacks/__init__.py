from .clr import LrFinder, SuperConvergence, WarmRestart

callback_maps = {
    #     "csv logger": CSVLogger,
    #     "csv_logger": CSVLogger,
    #     "csv": CSVLogger,
    #     "tensorboard": Tensorboard,
    #     "plateau": ReduceLROnPlateau,
    #     "early stopping": EarlyStopping,
    #     "early_stopping": EarlyStopping,
    #     "checkpoint": ModelCheckpoint,
    "super_convergence": SuperConvergence,
    "lr_finder": LrFinder,
    "warm_restart": WarmRestart,
}
