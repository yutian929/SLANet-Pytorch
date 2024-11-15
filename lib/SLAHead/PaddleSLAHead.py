# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
import numpy as np

class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
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

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output

class AttentionGRUCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        # print(f"Paddle Attention GRU Cell:\ninput_size={input_size}, hidden_size={hidden_size}, num_embeddings={num_embeddings}")
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size
        )

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)

        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


def get_para_bias_attr(l2_decay, k):
    if l2_decay > 0:
        regularizer = paddle.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = nn.initializer.Uniform(-stdv, stdv)
    else:
        regularizer = None
        initializer = None
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]




class HWAttention(nn.Layer):
    def __init__(
        self,
        head_dim=32,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // 3
        qkv = x.reshape([B, N, 3, C // self.head_dim, self.head_dim]).transpose(
            [2, 0, 3, 1, 4]
        )
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose([0, 1, 3, 2]) * self.scale
        attn = F.softmax(attn, -1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose([0, 2, 1]).reshape([B, N, C])
        return x


class SLAHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        use_attn=False,
        **kwargs,
    ):
        """
        @param in_channels: input shape
        @param hidden_size: hidden_size for RNN and Embedding
        @param out_channels: num_classes to rec
        @param max_text_length: max text pred
        """
        super().__init__()
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
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=hidden_size)
        weight_attr1_1, bias_attr1_1 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size
        )
        weight_attr1_2, bias_attr1_2 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size
        )
        self.structure_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1_2,
                bias_attr=bias_attr1_2,
            ),
            nn.Linear(
                hidden_size, out_channels, weight_attr=weight_attr, bias_attr=bias_attr
            ),
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
        weight_attr1, bias_attr1 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size
        )
        weight_attr2, bias_attr2 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size
        )
        self.loc_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1,
            ),
            nn.Linear(
                self.hidden_size,
                loc_reg_num,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs, targets=None):
        fea = inputs[-1]
        batch_size = fea.shape[0]
        if self.use_attn:
            fea = fea + self.cross_atten(fea)
        # reshape
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], -1])  # 1 x 96 x 16 x 16 → 1 x 96 x 256
        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)  # 1 x 256 x 96

        hidden = paddle.zeros((batch_size, self.hidden_size))
        structure_preds = paddle.zeros(  # 1 x 501 x 30
            (batch_size, self.max_text_length + 1, self.num_embeddings)
        )
        loc_preds = paddle.zeros(  # 1 x 501 x 8
            (batch_size, self.max_text_length + 1, self.loc_reg_num)
        )
        structure_preds.stop_gradient = True
        loc_preds.stop_gradient = True

        if self.training and targets is not None:
            structure = targets[0]
            max_len = targets[-2].max().astype("int32")
            for i in range(max_len + 1):
                hidden, structure_step, loc_step = self._decode(
                    structure[:, i], fea, hidden
                )
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
            structure_preds = structure_preds[:, : max_len + 1]
            loc_preds = loc_preds[:, : max_len + 1]
        else:  # infer
            structure_ids = paddle.zeros(
                (batch_size, self.max_text_length + 1), dtype="int32"
            )
            pre_chars = paddle.zeros(shape=[batch_size], dtype="int32")
            max_text_length = paddle.to_tensor(self.max_text_length)
            for i in range(max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(pre_chars, fea, hidden)  # 1. pre_chars=[0], fea=[1,256,96], hidden=[1,256] -> hidden=[1,256], structure_step=[1,50], loc_step=[1,8]
                pre_chars = structure_step.argmax(axis=1, dtype="int32")  # 1. pre_chars=[5]
                structure_preds[:, i, :] = structure_step  # 1. structure_preds=[1,501,50]
                loc_preds[:, i, :] = loc_step  # 1. loc_preds=[1,501,8]

                structure_ids[:, i] = pre_chars  # 1. structure_ids=[1,501]
                if (structure_ids == self.eos).any(-1).all():
                    break
        if not self.training:
            structure_preds = F.softmax(structure_preds[:, : i + 1])
            loc_preds = loc_preds[:, : i + 1]
        # breakpoint()
        return {"structure_probs": structure_preds, "loc_preds": loc_preds}

    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        # output shape is b * self.hidden_size
        (output, hidden), alpha = self.structure_attention_cell(
            hidden, features, emb_feature
        )

        # structure
        structure_step = self.structure_generator(output)
        # loc
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        input_ont_hot = F.one_hot(input_char, self.num_embeddings)
        return input_ont_hot

