import torch
import torch.nn as nn


def get_model(model, pretrained_path, freeze_feature, num_classes=2, act="sigmoid"):
    backbone = model(in_channels=1, num_classes=num_classes)

    if freeze_feature:
        print("freeze_feature")
        child = nn.Sequential(*list(backbone.children())[:-1])
        for param in child.parameters():
            param.requires_grad = False

    model = nn.Sequential()
    model.backbone = backbone

    if act == "sigmoid":
        model.act = nn.Sigmoid()
    elif act == "softmax":
        model.act = nn.Softmax(dim=1)

    if pretrained_path is not None:
        w = torch.load(pretrained_path)
        # print(model.load_state_dict(w["model_state_dict"]))
        print(model.load_state_dict(w, strict=False))
        print("pretrained_path:", pretrained_path)
        print("num param:", sum(p.numel() for p in model.parameters()))

    return model
