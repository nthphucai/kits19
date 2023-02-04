import torch.optim as optim

scheduler_maps = {
    "reduce_plateu": optim.lr_scheduler.ReduceLROnPlateau,
}
