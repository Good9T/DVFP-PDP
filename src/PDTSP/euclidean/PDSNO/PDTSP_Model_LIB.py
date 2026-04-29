import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _get_encoding(encoded_node, index_to_pick):
    # encoded_node shape: (batch, node, embedding)
    # index_to_pick shape: (batch, mt)
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_node.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    # shape: (batch, mt, embedding)
    picked_node = encoded_node.gather(dim=1, index=index_to_gather)
    # shape: (batch, mt, embedding)
    return picked_node

def reshape_by_heads(qkv, head_num):
    B, N, _ = qkv.shape
    return qkv.reshape(B, N, head_num, -1).transpose(1, 2)


def encoder_attention(
    q_n, k_n, v_n,
    q_p, k_p, v_p,
    q_d, k_d, v_d,
    head_num, qkv_dim, norm_factor
):
    batch, _, node_total, _ = q_n.shape
    pick_num = (node_total - 1) // 2

    v_d_addition = torch.cat([
        torch.zeros(batch, head_num, 1, qkv_dim, device=v_n.device),
        v_d,
        torch.zeros(batch, head_num, pick_num, qkv_dim, device=v_n.device)
    ], dim=2)

    v_p_addition = torch.cat([
        torch.zeros(batch, head_num, 1, qkv_dim, device=v_n.device),
        torch.zeros(batch, head_num, pick_num, qkv_dim, device=v_n.device),
        v_p
    ], dim=2)

    score_n = norm_factor * torch.matmul(q_n, k_n.transpose(2, 3))
    score_p_d1 = norm_factor * torch.sum(q_p * k_d, -1)
    score_p = norm_factor * torch.matmul(q_p, k_p.transpose(2, 3))
    score_p_d = norm_factor * torch.matmul(q_p, k_d.transpose(2, 3))
    score_d_p1 = norm_factor * torch.sum(q_d * k_p, -1)
    score_d = norm_factor * torch.matmul(q_d, k_d.transpose(2, 3))
    score_d_p = norm_factor * torch.matmul(q_d, k_p.transpose(2, 3))

    score_p_d1_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, device=score_n.device),
        score_p_d1,
        -torch.inf * torch.ones(batch, head_num, pick_num, device=score_n.device)
    ], dim=-1).unsqueeze(-1)

    score_p_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, pick_num, device=score_n.device),
        score_p,
        -torch.inf * torch.ones(batch, head_num, pick_num, pick_num, device=score_n.device)
    ], dim=2)

    score_p_d_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, pick_num, device=score_n.device),
        score_p_d,
        -torch.inf * torch.ones(batch, head_num, pick_num, pick_num, device=score_n.device)
    ], dim=2)

    score_d_p1_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, device=score_n.device),
        -torch.inf * torch.ones(batch, head_num, pick_num, device=score_n.device),
        score_d_p1,
    ], dim=-1).unsqueeze(-1)

    score_d_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, pick_num, device=score_n.device),
        -torch.inf * torch.ones(batch, head_num, pick_num, pick_num, device=score_n.device),
        score_d
    ], dim=2)

    score_d_p_addition = torch.cat([
        -torch.inf * torch.ones(batch, head_num, 1, pick_num, device=score_n.device),
        -torch.inf * torch.ones(batch, head_num, pick_num, pick_num, device=score_n.device),
        score_d_p
    ], dim=2)

    score = torch.cat([score_n, score_p_d1_addition, score_p_addition, score_p_d_addition,
                       score_d_p1_addition, score_d_addition, score_d_p_addition], dim=-1)

    attention = torch.softmax(score, dim=-1)
    sdpa = torch.matmul(attention[:, :, :, :node_total], v_n) \
           + attention[:, :, :, node_total].unsqueeze(-1) * v_d_addition \
           + torch.matmul(attention[:, :, :, 1 + node_total:1 + node_total + pick_num], v_p) \
           + torch.matmul(attention[:, :, :, 1 + node_total + pick_num:node_total * 2], v_d) \
           + attention[:, :, :, node_total * 2].unsqueeze(-1) * v_p_addition \
           + torch.matmul(attention[:, :, :, 1 + node_total * 2:1 + node_total * 2 + pick_num], v_d) \
           + torch.matmul(attention[:, :, :, 1 + node_total * 2 + pick_num:], v_p)

    return sdpa

def decoder_attention(q_n, k_n, v_n, q_p, k_p, q_d, k_d, rank3_mask=None):
    B, H, M, _ = q_n.shape
    N = k_n.shape[2]
    P = k_p.shape[2]

    score_n = torch.matmul(q_n, k_n.transpose(-2, -1)) / math.sqrt(k_n.shape[-1])
    score_p = torch.matmul(q_p, k_p.transpose(-2, -1)) / math.sqrt(k_p.shape[-1])
    score_d = torch.matmul(q_d, k_d.transpose(-2, -1)) / math.sqrt(k_d.shape[-1])

    if rank3_mask is not None:
        score_n += rank3_mask[:, None, :, :]
        score_p += rank3_mask[:, None, :, 1:1 + P]
        score_d += rank3_mask[:, None, :, 1 + P:]

    score = torch.cat([score_n, score_p, score_d], dim=-1)
    w = torch.softmax(score, dim=-1)
    out = torch.matmul(w[..., :N], v_n)
    return out.transpose(1, 2).flatten(2)

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = nn.InstanceNorm1d(model_params['embedding_dim'], affine=True)

    def forward(self, x1, x2):
        added = x1 + x2
        return self.norm(added.transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        emb_dim = model_params['embedding_dim']
        ff_dim = model_params['ff_hidden_dim']
        self.fc1 = nn.Linear(emb_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))