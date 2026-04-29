import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_encoding(encoded_node, index_to_pick):
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_node.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    picked_node = encoded_node.gather(dim=1, index=index_to_gather)
    return picked_node

def reshape_by_heads(qkv, head_num):
    B, N, _ = qkv.shape
    return qkv.reshape(B, N, head_num, -1).transpose(1, 2)

def multi_head_attention(q, k, v, rank3_ninf_mask=None):
    B, H, N, K = q.shape
    score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(K, dtype=torch.float))

    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :]

    weight = F.softmax(score, dim=-1)
    out = torch.matmul(weight, v).transpose(1, 2)
    return out.reshape(B, N, -1)

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