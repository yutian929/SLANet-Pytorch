import torch
import torch.nn as nn
from .torch_build import build_transform, build_backbone, build_neck, build_head

DEBUG = True
if DEBUG:
    import numpy as np
    import os
    
    SAVE_PREFIX = "./debug_data/pytorch/"
    os.makedirs(SAVE_PREFIX, exist_ok=True)

    def save_tensor(tensor, filename):
        if isinstance(tensor, torch.Tensor):
            np.save(f"{filename}.npy", tensor.detach().cpu().numpy())
        else:
            print(f"Failed to save {filename}: Provided data is not a Tensor.")

    def save_numpy_array(output, filename):
        # Prefix for saving files
        filename = SAVE_PREFIX + filename
        # Handle different types of outputs
        if isinstance(output, torch.Tensor):
            save_tensor(output, filename)
        elif isinstance(output, (list, tuple)):  # Handle list or tuple of tensors
            for idx, tensor in enumerate(output):
                save_tensor(tensor, f"{filename}_{idx}")  # Save each tensor with an index
        elif isinstance(output, dict):  # Handle dictionary of tensors
            for key, tensor in output.items():
                save_tensor(tensor, f"{filename}_{key}")  # Save each tensor with its key
        else:
            print(f"Failed to save output: Unsupported data type {type(output)}.")
    


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        # if you make model differently, you can use transfrom in det and cls
        if "Transform" not in config or config["Transform"] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        if "Backbone" not in config or config["Backbone"] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]["in_channels"] = in_channels
            self.backbone = build_backbone(config["Backbone"])
            in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if "Neck" not in config or config["Neck"] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_neck(config["Neck"])
            in_channels = self.neck.out_channels

        # # build head, head is need for det, rec and cls
        if "Head" not in config or config["Head"] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]["in_channels"] = in_channels
            self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        if DEBUG:
            save_numpy_array(x, "input")
        # x 输入为 1x3x488x488
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if DEBUG:
            save_numpy_array(x, "backbone_output")
        # 经过LCNet之后，记录四部分的输出值，各自Shape如下：
        # 1 x 64 x 122 x 122
        # 1 x 128 x 61 x 61
        # 1 x 256 x 31 x 31
        # 1 x 512 x 16 x 16
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if DEBUG:
            save_numpy_array(x, "neck_output")
        # 经过CSPLayer + PAN结构，包括（top-down和down-top两个结构），各自shape如下：
        # 1 x 96 x 122 x 122
        # 1 x 96 x 61 x 61
        # 1 x 96 x 31 x 31
        # 1 x 96 x 16 x 16
        # 取 1 x 96 x 16 x 16这个特征，进入后续基于注意力机制的层
        if self.use_head:
            x = self.head(x)
            # for multi head, save ctc neck out for udml
            if isinstance(x, dict) and "ctc_neck" in x.keys():
                y["neck_out"] = x["ctc_neck"]
                y["head_out"] = x
            elif isinstance(x, dict):
                y.update(x)
            else:
                y["head_out"] = x
            final_name = "head_out"
        if DEBUG:
            save_numpy_array(x, "head_output")
        if self.return_all_feats:
            if self.training:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x