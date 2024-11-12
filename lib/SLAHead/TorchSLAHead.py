# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import torch
from torch import nn
from torch.nn import functional as F


class SLAHead(nn.Module):
    def __init__(self, in_channels=96, is_train=False, hidden_size=256, out_channels=50, max_text_length=500, loc_reg_num=4) -> None:
        super().__init__()
        self.max_text_length = max_text_length
        self.hidden_size = hidden_size
        self.loc_reg_num = loc_reg_num
        self.out_channels = out_channels
        self.num_embeddings = self.out_channels
        self.is_train = is_train
        self.eos = self.num_embeddings - 1

        self.structure_attention_cell = AttentionGRUCell(in_channels,
                                                         self.hidden_size,
                                                         self.num_embeddings)

        self.structure_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_channels)
        )

        self.loc_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.loc_reg_num)
        )

    def forward(self, fea):
        fea = fea[-1]
        batch_size = fea.shape[0]

        # 1 x 96 x 16 x 16 → 1 x 96 x 256
        fea = torch.reshape(fea, [fea.shape[0], fea.shape[1], -1])

        # 1 x 256 x 96
        fea = fea.permute(0, 2, 1)

        # infer 1 x 501 x 30
        structure_preds = torch.zeros(batch_size, self.max_text_length + 1,
                                      self.num_embeddings)
        # 1 x 501 x 8
        loc_preds = torch.zeros(batch_size, self.max_text_length + 1,
                                self.loc_reg_num)

        hidden = torch.zeros(batch_size, self.hidden_size)
        pre_chars = torch.zeros(batch_size, dtype=torch.int64)

        structure_ids = torch.zeros(
            (batch_size, self.max_text_length+1), dtype=torch.int32
        )
        loc_step, structure_step = None, None
        for i in range(self.max_text_length + 1):
            hidden, structure_step, loc_step = self._decode(pre_chars,  # 1. same
                                                            fea, hidden)
            pre_chars = structure_step.argmax(dim=1)
            structure_preds[:, i, :] = structure_step
            loc_preds[:, i, :] = loc_step

            structure_ids[:, i] = pre_chars
            if (structure_ids == self.eos).any(-1).all():
                break

        if not self.is_train:
            # structure_preds = F.softmax(structure_preds, dim=-1)
            structure_preds = F.softmax(structure_preds[:, : i + 1])
            loc_preds = loc_preds[:, : i + 1]
        # structure_preds: 1 x 501 x 30
        # loc_preds: 1 x 501 x 8
        # breakpoint()
        return structure_preds, loc_preds

    def _decode(self, pre_chars, features, hidden):
        emb_features = F.one_hot(pre_chars, num_classes=self.num_embeddings)
        (output, hidden), alpha = self.structure_attention_cell(hidden,
                                                                features,
                                                                emb_features)
        structure_step = self.structure_generator(output)
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embedding) -> None:
        super().__init__()

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.gru = nn.GRU(input_size=input_size + num_embedding,
                          hidden_size=hidden_size,)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # 这里实现参考论文https://arxiv.org/pdf/1704.03549.pdf
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = F.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.concat([context, char_onehots], 1)

        cur_hidden = self.gru(concat_context, prev_hidden)
        return cur_hidden, alpha


class SLALoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.structure_weight = 1.0
        self.loc_weight = 2.0
        self.eps = 1e-12

    def forward(self, pred):
        structure_probs = pred[0]
        structure_probs = structure_probs.permute(0, 2, 1)
        # 1 x 30 x 501

        # 1 x 501
        structure_target = torch.empty(1, 501, dtype=torch.long).random_(30)
        structure_loss = self.loss_func(structure_probs, structure_target)
        structure_loss = structure_loss * self.structure_weight

        loc_preds = pred[1]  # 1 x 501 x 4
        loc_targets = torch.randn(1, 501, 4)
        loc_target_mask = torch.randn(1, 501, 1)

        loc_loss = F.smooth_l1_loss(loc_preds * loc_target_mask,
                                    loc_targets * loc_target_mask,
                                    reduction='mean')
        loc_loss *= self.loc_weight
        loc_loss = loc_loss / (loc_target_mask.sum() + self.eps)

        total_loss = structure_loss + loc_loss
        return total_loss


if __name__ == '__main__':
    # x 输入为 1x3x488x488
    # 经过LCNet之后，记录四部分的输出值，各自Shape如下：
    # 1 x 64 x 122 x 122
    # 1 x 128 x 61 x 61
    # 1 x 256 x 31 x 31
    # 1 x 512 x 16 x 16

    # 经过CSPLayer + PAN结构，包括（top-down和down-top两个结构），各自shape如下：
    # 1 x 96 x 122 x 122
    # 1 x 96 x 61 x 61
    # 1 x 96 x 31 x 31
    # 1 x 96 x 16 x 16

    # 取 1 x 96 x 16 x 16这个特征，进入后续基于注意力机制的层
    x = torch.randn(1, 96, 16, 16)

    model = SLAHead()
    loss_func = SLALoss()

    y = model(x)

    loss = loss_func(y)

    print(y[0].shape)
    print(y[1].shape)

