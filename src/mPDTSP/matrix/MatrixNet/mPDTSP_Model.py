import torch
import torch.nn as nn
import torch.nn.functional as F

from mPDTSP_Model_LIB import AddAndInstanceNormalization, FeedForward, MixedScoreAttention, _get_encoding, reshape_by_heads


class mPDTSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = mPDTSP_Encoder(**model_params)
        self.decoder = mPDTSP_Decoder(**model_params)
        self.embedding_dim = int(self.model_params['embedding_dim'])
        self.seed_num = int(self.model_params['one_hot_seed_num'])

        self.encoded_row = None
        self.encoded_col = None

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        # shape: (batch, node, node)
        batch_size = problems.size(0)
        node_size = problems.size(1)

        embedding_dim = self.embedding_dim

        row_emb = torch.zeros(size=(batch_size, node_size, embedding_dim), device=problems.device)
        col_emb = torch.zeros(size=(batch_size, node_size, embedding_dim), device=problems.device)

        customer_size = (node_size - 1) // 2  # 50

        rand = torch.rand(batch_size, customer_size, device=problems.device)
        batch_rand_perm = rand.argsort(dim=1) + 1
        rand_idx = torch.zeros(batch_size, node_size, dtype=torch.long, device=problems.device)

        # depot
        rand_idx[:, 0] = 0

        # P
        rand_idx[:, 1: 1 + customer_size] = batch_rand_perm

        # D
        rand_idx[:, 1 + customer_size:] = batch_rand_perm + customer_size

        batch_idx = torch.arange(batch_size, device=problems.device)[:, None].expand(batch_size, node_size)
        node_idx = torch.arange(node_size, device=problems.device)[None, :].expand(batch_size, node_size)
        col_emb[batch_idx, node_idx, rand_idx] = 1

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(batch_size, pomo_size)
            encoded_first_row = _get_encoding(self.encoded_row, selected)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            all_probs = self.decoder(encoded_current_row, mask=state.state_mask)

            if self.training:
                while True:
                    with torch.no_grad():
                        selected = all_probs.flatten(0, 1).multinomial(1).view(batch_size, pomo_size)
                    prob = all_probs[state.BATCH_IDX, state.POMO_IDX, selected]
                    if (prob != 0).all():
                        break
            else:
                selected = all_probs.argmax(dim=2)
                prob = None

        return selected, prob


class mPDTSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        layer_num = int(model_params["encoder_layer_num"])
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(layer_num)])

    def forward(self, row_emb, col_emb, problem):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, problem)
        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_block = EncodingBlock(**model_params)
        self.col_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, problem):
        row_out = self.row_block(row_emb, col_emb, problem)
        col_out = self.col_block(col_emb, row_emb, problem.transpose(1, 2))
        return row_out, col_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.emb_dim = int(model_params["embedding_dim"])
        self.head_num = int(model_params["head_num"])
        self.qkv_dim = int(model_params["qkv_dim"])

        self.Wq = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)

        self.attention = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.emb_dim)

        self.norm1 = AddAndInstanceNormalization(**model_params)
        self.ff = FeedForward(**model_params)
        self.norm2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row, col, cost_matrix):
        q = reshape_by_heads(self.Wq(row), self.head_num)
        k = reshape_by_heads(self.Wk(col), self.head_num)
        v = reshape_by_heads(self.Wv(col), self.head_num)

        out = self.attention(q, k, v, cost_matrix)
        out = self.multi_head_combine(out)

        out = self.norm1(row, out)
        out = self.norm2(out, self.ff(out))
        return out


class mPDTSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.emb_dim = int(model_params["embedding_dim"])
        self.head_num = int(model_params["head_num"])
        self.qkv_dim = int(model_params["qkv_dim"])

        self.Wq0 = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq1 = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.emb_dim)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None

    def set_kv(self, encoded_col):
        self.k = reshape_by_heads(self.Wk(encoded_col), self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_col), self.head_num)
        self.single_head_key = encoded_col.transpose(1, 2)

    def set_q1(self, first_row):
        self.q1 = reshape_by_heads(self.Wq1(first_row), self.head_num)

    def _attention(self, q, k, v, mask):
        batch_size, head_num, n = q.shape[:3]
        node_num = k.shape[2]
        score = torch.matmul(q, k.transpose(2, 3)) / self.model_params["sqrt_qkv_dim"]

        if mask is not None:
            score = score + mask[:, None, :, :].expand(batch_size, head_num, n, node_num)

        weights = score.softmax(dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).flatten(2)
        return out

    def forward(self, q0, mask):
        q0 = reshape_by_heads(self.Wq0(q0), self.head_num)
        q = self.q1 + q0

        out = self._attention(q, self.k, self.v, mask)
        out = self.multi_head_combine(out)

        score = torch.matmul(out, self.single_head_key)
        score = score / self.model_params["sqrt_embedding_dim"]
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        score = score + mask

        return F.softmax(score, dim=-1)