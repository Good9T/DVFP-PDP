import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mPDTSP_Model_LIB import _get_encoding, reshape_by_heads, encoder_attention, decoder_attention, AddAndInstanceNormalization, FeedForward

class mPDTSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)

        # 固定 POMO 风格变量名
        self.encoded_node = None

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        batch_size = problems.size(0)
        node_size = problems.size(1)
        customer_size = (node_size - 1) // 2
        angle = torch.rand(batch_size) * 2 * torch.pi
        cos, sin = torch.cos(angle), torch.sin(angle)

        rot_matrix = torch.stack([
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin, cos], dim=1),
        ], dim=1)
        center = torch.tensor([0.5, 0.5])[None, None, :]
        coords = problems[..., :2]
        others = problems[..., 2:]

        coords_rot = torch.bmm(coords - center, rot_matrix) + center
        problems = torch.cat([coords_rot, others], dim=-1)
        depot = problems[:, 0:1, :]
        pick = problems[:, 1:1 + customer_size, :]
        deli = problems[:, 1 + customer_size:, :]

        self.encoded_node = self.encoder(depot, pick, deli)
        self.decoder.set_kv(self.encoded_node)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.POMO_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.zeros(batch_size, pomo_size, dtype=torch.long)
            prob = torch.ones(batch_size, pomo_size)

        elif state.selected_count == 1:
            selected = torch.arange(1, pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(batch_size, pomo_size)

        else:
            encoded_last = _get_encoding(self.encoded_node, state.current_node)
            prob = self.decoder(encoded_last, state.state_mask)

            if self.training:
                selected = prob.flatten(0, 1).multinomial(1).view(batch_size, pomo_size)
                prob = prob[state.BATCH_IDX, state.POMO_IDX, selected]
            else:
                selected = prob.argmax(dim=2)
                prob = None

        return selected, prob


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = int(self.model_params['embedding_dim'])
        self.encoder_layer_num = int(self.model_params['encoder_layer_num'])

        self.embedding_depot = nn.Linear(3, self.embedding_dim)
        self.embedding_pick = nn.Linear(3, self.embedding_dim)
        self.embedding_delivery = nn.Linear(3, self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])

    def forward(self, depot_x_y, pick_x_y, delivery_x_y):
        embedding_depot = self.embedding_depot(depot_x_y)
        embedded_pick = self.embedding_pick(pick_x_y)
        embedded_delivery = self.embedding_delivery(delivery_x_y)
        embedded_node = torch.cat((embedding_depot, embedded_pick, embedded_delivery), dim=1)

        for layer in self.layers:
            embedded_node = layer(embedded_node)

        return embedded_node


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = int(self.model_params['embedding_dim'])
        self.head_num = int(self.model_params['head_num'])
        self.qkv_dim = int(self.model_params['qkv_dim'])

        self.Wq_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = AddAndInstanceNormalization(**model_params)
        self.ff = FeedForward(**model_params)
        self.norm2 = AddAndInstanceNormalization(**model_params)
        self.norm_factor = 1 / math.sqrt(self.qkv_dim)

    def forward(self, embedded_node):
        batch = embedded_node.size(0)
        node = embedded_node.size(1)
        pick = (node - 1) // 2

        q_n = reshape_by_heads(self.Wq_n(embedded_node), head_num=self.head_num)
        k_n = reshape_by_heads(self.Wk_n(embedded_node), head_num=self.head_num)
        v_n = reshape_by_heads(self.Wv_n(embedded_node), head_num=self.head_num)

        embedded_pick = embedded_node[:, 1:1 + pick, :].contiguous()
        embedded_delivery = embedded_node[:, 1 + pick:, :].contiguous()

        q_p = reshape_by_heads(self.Wq_p(embedded_pick), head_num=self.head_num)
        k_p = reshape_by_heads(self.Wk_p(embedded_pick), head_num=self.head_num)
        v_p = reshape_by_heads(self.Wv_p(embedded_pick), head_num=self.head_num)

        q_d = reshape_by_heads(self.Wq_d(embedded_delivery), head_num=self.head_num)
        k_d = reshape_by_heads(self.Wk_d(embedded_delivery), head_num=self.head_num)
        v_d = reshape_by_heads(self.Wv_d(embedded_delivery), head_num=self.head_num)

        # ========== 调用抽出去的函数 ==========
        sdpa = encoder_attention(
            q_n, k_n, v_n,
            q_p, k_p, v_p,
            q_d, k_d, v_d,
            self.head_num, self.qkv_dim, self.norm_factor
        )

        out = self.multi_head_combine(sdpa.permute(0, 2, 1, 3).flatten(2))
        out1 = self.norm1(embedded_node, out)
        out2 = self.ff(out1)
        out3 = self.norm2(out1, out2)
        return out3


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = int(self.model_params['embedding_dim'])
        self.head_num = int(self.model_params['head_num'])
        self.qkv_dim = int(self.model_params['qkv_dim'])
        self.clip = int(self.model_params['logit_clipping'])

        self.Wq_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.q_n = None
        self.q_d = None
        self.q_p = None

        self.node_key = None
        self.k_n = self.v_n = self.k_p = self.k_d = None
        self.customer_size = None

    def set_kv(self, encoded_node):
        self.customer_size = (encoded_node.size(1) - 1) // 2
        enc_p = encoded_node[:, 1:1 + self.customer_size]
        enc_d = encoded_node[:, 1 + self.customer_size:]

        self.k_n = reshape_by_heads(self.Wk_n(encoded_node), self.head_num)
        self.v_n = reshape_by_heads(self.Wv_n(encoded_node), self.head_num)
        self.k_p = reshape_by_heads(self.Wk_p(enc_p), self.head_num)
        self.k_d = reshape_by_heads(self.Wk_d(enc_d), self.head_num)
        self.node_key = encoded_node.transpose(1, 2)

    def forward(self, enc_last, mask):
        self.q_n = reshape_by_heads(self.Wq_n(enc_last), self.head_num)
        self.q_p = reshape_by_heads(self.Wq_p(enc_last), self.head_num)
        self.q_d = reshape_by_heads(self.Wq_d(enc_last), self.head_num)

        attn = decoder_attention(self.q_n, self.k_n, self.v_n, self.q_p, self.k_p, self.q_d, self.k_d, mask)
        score = self.multi_head_combine(attn)
        score = torch.matmul(score, self.node_key) / math.sqrt(self.embedding_dim)
        score = self.clip * torch.tanh(score) + mask
        return F.softmax(score, dim=2)


