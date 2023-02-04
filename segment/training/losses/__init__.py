from .loss import BCE, WBCE, Dice_BCE, DiceLoss, FocalDiceLoss, FocalLoss

loss_maps = {
    "dice_bce": Dice_BCE,
    "dice": DiceLoss,
    "bce": BCE,
    "wbce": WBCE,
    "focal": FocalLoss,
    "focal_dice": FocalDiceLoss,
}
