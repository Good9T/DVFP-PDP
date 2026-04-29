import torch
import torch.nn as nn
import torch.nn.functional as F
from PDTSP_Model_LIB import _get_encoding, reshape_by_heads, multi_head_attention, AddAndInstanceNormalization, FeedForward


class PDTSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = PDTSP_Encoder(**model_params)
        self.decoder = PDTSP_Decoder(**model_params)
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


class PDTSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.layer_num = int(model_params['encoder_layer_num'])

        self.emb_depot = nn.Linear(2, self.embedding_dim)
        self.emb_pick = nn.Linear(2, self.embedding_dim)
        self.emb_deli = nn.Linear(2, self.embedding_dim)
        self.encoded_node = None
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.layer_num)])

    def forward(self, depot_x_y, pick_x_y, delivery_x_y):
        embedding_depot = self.emb_depot(depot_x_y)
        embedded_pick = self.emb_pick(pick_x_y)
        embedded_delivery = self.emb_deli(delivery_x_y)
        embedded_node = torch.cat((embedding_depot, embedded_pick, embedded_delivery), dim=1)

        for layer in self.layers:
            embedded_node = layer(embedded_node)

        return embedded_node


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.head_num = int(model_params['head_num'])
        self.qkv_dim = int(model_params['qkv_dim'])

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = AddAndInstanceNormalization(**model_params)
        self.ff = FeedForward(**model_params)
        self.norm2 = AddAndInstanceNormalization(**model_params)

    def forward(self, x):
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)

        attn_out = multi_head_attention(q, k, v)
        attn_out = self.multi_head_combine(attn_out)

        x = self.norm1(x, attn_out)
        x = self.norm2(x, self.ff(x))
        return x


class PDTSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.head_num = int(model_params['head_num'])
        self.qkv_dim = int(model_params['qkv_dim'])
        self.clip = float(model_params['logit_clipping'])
        self.sqrt_emb = float(model_params['sqrt_embedding_dim'])

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.k = None
        self.v = None
        self.single_key = None

    def set_kv(self, enc_node):
        self.k = reshape_by_heads(self.Wk(enc_node), self.head_num)
        self.v = reshape_by_heads(self.Wv(enc_node), self.head_num)
        self.single_key = enc_node.transpose(1, 2)

    def forward(self, enc_last, mask):
        q = reshape_by_heads(self.Wq(enc_last), self.head_num)

        attn_out = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=mask)
        attn_out = self.multi_head_combine(attn_out)

        score = torch.matmul(attn_out, self.single_key) / self.sqrt_emb
        score = self.clip * torch.tanh(score)
        score = score + mask

        return F.softmax(score, dim=-1)
