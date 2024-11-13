import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size
        )

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden).unsqueeze(1)

        res = batch_H_proj + prev_hidden_proj
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.transpose(1, 2)
        context = torch.bmm(alpha, batch_H).squeeze(1)
        concat_context = torch.cat([context, char_onehots], dim=1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha

class HWAttention(nn.Module):
    def __init__(
        self,
        head_dim=32,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super(HWAttention, self).__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // 3
        qkv = x.view(B, N, 3, C // self.head_dim, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return x

class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0
    ):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HWAttention(head_dim=dim // num_heads)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SLAHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=8,
        fc_decay=0.0,
        use_attn=False,
        **kwargs,
    ):
        super(SLAHead, self).__init__()
        in_channels = in_channels[-1]
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num
        self.eos = self.num_embeddings - 1

        # structure
        self.structure_attention_cell = AttentionGRUCell(
            in_channels, hidden_size, self.num_embeddings
        )
        self.structure_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(hidden_size, out_channels),
        )
        dpr = np.linspace(0, 0.1, 2)

        self.use_attn = use_attn
        if use_attn:
            layer_list = [
                Block(
                    in_channels,
                    num_heads=2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(2)
            ]
            self.cross_atten = nn.Sequential(*layer_list)
        # loc
        self.loc_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, loc_reg_num),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def forward(self, inputs, targets=None):
        fea = inputs[-1]
        batch_size = fea.shape[0]
        if self.use_attn:
            fea = fea + self.cross_atten(fea)
        # reshape
        fea = fea.reshape(fea.shape[0], fea.shape[1], -1)  # B x C x H x W -> B x C x N
        fea = fea.permute(0, 2, 1)  # B x N x C

        device = fea.device
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        structure_preds = torch.zeros(
            batch_size, self.max_text_length + 1, self.num_embeddings, device=device
        )
        loc_preds = torch.zeros(
            batch_size, self.max_text_length + 1, self.loc_reg_num, device=device
        )
        structure_preds.requires_grad = False
        loc_preds.requires_grad = False

        if self.training and targets is not None:
            structure = targets[0]
            max_len = targets[-2].max().int()
            for i in range(max_len + 1):
                hidden, structure_step, loc_step = self._decode(
                    structure[:, i], fea, hidden
                )
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
            structure_preds = structure_preds[:, : max_len + 1]
            loc_preds = loc_preds[:, : max_len + 1]
        else:  # inference
            structure_ids = torch.zeros(
                batch_size, self.max_text_length + 1, dtype=torch.long, device=device
            )
            pre_chars = torch.zeros(batch_size, dtype=torch.long, device=device)
            max_text_length = self.max_text_length
            for i in range(max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(
                    pre_chars, fea, hidden
                )
                pre_chars = structure_step.argmax(dim=1).long()
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
                structure_ids[:, i] = pre_chars
                if (structure_ids == self.eos).any(dim=-1).all():
                    break
            structure_preds = structure_preds[:, : i + 1]
            loc_preds = loc_preds[:, : i + 1]
        if not self.training:
            structure_preds = F.softmax(structure_preds, dim=-1)
        return {"structure_probs": structure_preds, "loc_preds": loc_preds}

    def _decode(self, pre_chars, features, hidden):
        emb_feature = self.emb(pre_chars)
        hidden, alpha = self.structure_attention_cell(
            hidden, features, emb_feature
        )
        # structure
        structure_step = self.structure_generator(hidden)
        # loc
        loc_step = self.loc_generator(hidden)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        input_one_hot = F.one_hot(
            input_char, num_classes=self.num_embeddings
        ).float()
        return input_one_hot

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.uniform_(m.weight, -stdv, stdv)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -stdv, stdv)

# Unit Test
def test_SLAHead():
    # Input tensor
    x = torch.ones(1, 96, 16, 16)
    inputs = [x]  # Since inputs is a list in forward function

    # Create the model
    model = SLAHead(in_channels=[96], hidden_size=256, out_channels=50, max_text_length=500, loc_reg_num=8)

    # Set model to eval mode
    model.eval()

    # Run the model
    with torch.no_grad():
        outputs = model(inputs)

    # Print the outputs
    print("Structure probabilities shape:", outputs['structure_probs'].shape)
    print("Location predictions shape:", outputs['loc_preds'].shape)

if __name__ == "__main__":
    test_SLAHead()
