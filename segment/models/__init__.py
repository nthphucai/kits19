from .unet3D import BasicUnet3D, DyUnetRes3D, UnetRes3D_v1, UnetRes3D_v2

model_maps = {
    "BasicUnet": BasicUnet3D,
    "DyUnetRes": DyUnetRes3D,
    "UnetRes_v1": UnetRes3D_v1,
    "UnetRes_v2": UnetRes3D_v2,
}
