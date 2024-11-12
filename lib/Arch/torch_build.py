def build_transform(cfg):
    raise NotImplementedError

def build_backbone(cfg):
    from ..PPLCNet.TorchPPLCNet import PPLCNet
    return PPLCNet(in_channels=3, scale=1.0, pretrained=False)

def build_neck(cfg):
    from ..CSPPAN.TorchCSPPAN import CSPPAN
    return CSPPAN(in_channels=[64,128,256,512], out_channels=96)

def build_head(cfg):
    from ..SLAHead.TorchSLAHead import SLAHead
    return SLAHead(in_channels=96, is_train=False, hidden_size=256, out_channels=50, max_text_length=500, loc_reg_num=8)
