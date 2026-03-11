import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)

    def forward(self, input):
        # input: (batch, problem, embedding)
        transposed = input.transpose(1, 2)
        normalized = self.normalizer(transposed)
        return normalized.transpose(1, 2)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, q, k, v, attn_mask=None):
        # q: (batch, head_num, N, qkv_dim)
        # k: (batch, head_num, N, qkv_dim)
        # v: (batch, head_num, N, qkv_dim)
        B, head_num, n, key_dim = q.size()
        input_s = k.size(2)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[:, None, None, :].expand(B, head_num, n, input_s)
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None, :, :].expand(B, head_num, n, input_s)

        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / math.sqrt(key_dim)

        if attn_mask is not None:
            score_scaled = score_scaled + attn_mask

        weights = F.softmax(score_scaled, dim=-1)
        out = torch.matmul(weights, v)
        # (batch, head_num, n, qkv_dim) -> (batch, n, head_num * qkv_dim)
        out = out.transpose(1, 2).reshape(B, n, head_num * key_dim)
        return out


class MHABlock(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.sdpa = ScaledDotProductAttention(embedding_dim)

    def forward(self, input1):
        B, N, E = input1.size()
        q = self.Wq(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        k = self.Wk(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        v = self.Wv(input1).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        out_concat = self.sdpa(q, k, v)
        return self.multi_head_combine(out_concat)


class FFBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        ff_size = embedding_dim * 2
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_size * 2),  # *2 for SwiGLU
            SwiGLU(),
            nn.Linear(ff_size, embedding_dim),
        )

    def forward(self, input1):
        return self.feed_forward(input1)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim):
        super(EncoderLayer, self).__init__()
        self.mha_block = MHABlock(embedding_dim, head_num, qkv_dim)
        self.norm1 = Normalization(embedding_dim)
        self.ff_block = FFBlock(embedding_dim)
        self.norm2 = Normalization(embedding_dim)

    def forward(self, x):
        attn_out = self.mha_block(x)
        x = self.norm1(attn_out + x)
        ff_out = self.ff_block(x)
        x = self.norm2(x + ff_out)
        return x


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, num_layers=4, num_heads=4, qkv_dim=32):
        super(AttentionEncoder, self).__init__()
        self.input_embedder = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, qkv_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (B, N, input_dim) road features per edge
        returns: (B, N, embedding_dim)
        """
        out = self.input_embedder(x)
        for layer in self.layers:
            out = layer(out)
        return out


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim=128, cav_feature_dim=11, num_heads=4, qkv_dim=32):
        super(AttentionDecoder, self).__init__()
        self.head_num = num_heads
        self.qkv_dim = qkv_dim
        self.embedding_dim = embedding_dim

        query_dim = embedding_dim + cav_feature_dim

        self.Wq_first = nn.Linear(query_dim, num_heads * qkv_dim, bias=False)
        self.Wq = nn.Linear(query_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)
        self.sdpa = ScaledDotProductAttention(embedding_dim)

        self.k = None
        self.v = None
        self.q_first = 0
        self.single_head_key = None

    def set_kv(self, encoding):
        """Cache encoder output for K, V computation."""
        B, N, _ = encoding.shape
        self.k = self.Wk(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        self.v = self.Wv(encoding).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        self.single_head_key = encoding.transpose(1, 2)

    def set_q_first_node(self, first_node_emb, cav_features):
        """
        Cache the first node query (called once when CAV departs).
        first_node_emb: (B, 1, embedding_dim)
        cav_features: (B, cav_feature_dim)
        """
        query_input = torch.cat([first_node_emb, cav_features.unsqueeze(1).expand_as(
            first_node_emb[:, :, :cav_features.size(-1)].expand(-1, -1, cav_features.size(-1)))], dim=-1)
        B, N, _ = query_input.shape
        self.q_first = self.Wq_first(query_input).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)

    def forward(self, cur_node_emb, cav_features, mask=None):
        """
        cur_node_emb: (B, 1, embedding_dim)
        cav_features: (B, cav_feature_dim)
        returns: (B, 1, embedding_dim)
        """
        cav_expanded = cav_features.unsqueeze(1).expand(-1, cur_node_emb.size(1), -1)
        query_input = torch.cat([cur_node_emb, cav_expanded], dim=-1)

        B, N, _ = query_input.shape
        q_current = self.Wq(query_input).view(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        q = self.q_first + q_current

        out_concat = self.sdpa(q, self.k, self.v, mask)
        mh_attn_out = self.multi_head_combine(out_concat)
        return mh_attn_out


class PolicyHead(nn.Module):
    def __init__(self, embedding_dim=128, action_dim=3, C=10):
        super(PolicyHead, self).__init__()
        self.C = C
        self.embedding_dim = embedding_dim
        self.action_proj = nn.Linear(embedding_dim, action_dim)

    def forward(self, context, action_mask=None):
        """
        context: (B, 1, embedding_dim)
        action_mask: (B, action_dim) with 1=available, 0=unavailable
        returns: (B, action_dim) action probabilities
        """
        logits = self.action_proj(context.squeeze(1))
        score_scaled = logits / math.sqrt(self.embedding_dim)
        score_clipped = self.C * torch.tanh(score_scaled)
        if action_mask is not None:
            score_clipped = score_clipped.masked_fill(action_mask == 0, float('-inf'))
        probs = F.softmax(score_clipped, dim=-1)
        return probs


class ValueHead(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ValueHead, self).__init__()
        inner_size = embedding_dim
        self.val = nn.Sequential(
            nn.Linear(embedding_dim, inner_size * 2),  # *2 for SwiGLU
            SwiGLU(),
            nn.Linear(inner_size, 1),
        )

    def forward(self, context):
        """
        context: (B, 1, embedding_dim)
        returns: (B, 1) scalar value
        """
        return self.val(context.squeeze(1))


class AlphaRouterModel(nn.Module):
    def __init__(self, road_feature_dim=12, cav_feature_dim=11, action_dim=3,
                 embedding_dim=128, encoder_layers=4, num_heads=4,
                 qkv_dim=32, num_edges=192, C=10):
        super(AlphaRouterModel, self).__init__()
        self.road_feature_dim = road_feature_dim
        self.cav_feature_dim = cav_feature_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.num_edges = num_edges

        self.encoder = AttentionEncoder(road_feature_dim, embedding_dim, encoder_layers, num_heads, qkv_dim)
        self.decoder = AttentionDecoder(embedding_dim, cav_feature_dim, num_heads, qkv_dim)
        self.policy_head = PolicyHead(embedding_dim, action_dim, C)
        self.value_head = ValueHead(embedding_dim)

        self.encoding = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, state, action_mask=None, cur_edge_idx=None):
        """
        state: (B, flat_state_dim) = (B, num_edges * road_feature_dim + cav_feature_dim)
        action_mask: (B, action_dim)
        cur_edge_idx: (B,) index of the current edge
        returns: probs (B, action_dim), value (B, 1)
        """
        B = state.shape[0]
        road_flat = state[:, :self.num_edges * self.road_feature_dim]
        cav_features = state[:, self.num_edges * self.road_feature_dim:]

        road_features = road_flat.view(B, self.num_edges, self.road_feature_dim)

        self.encoding = self.encoder(road_features)
        self.decoder.set_kv(self.encoding)

        if cur_edge_idx is not None:
            idx = cur_edge_idx.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            cur_node_emb = torch.gather(self.encoding, 1, idx)
        else:
            cur_node_emb = self.encoding.mean(dim=1, keepdim=True)

        self.decoder.set_q_first_node(cur_node_emb, cav_features)

        context = self.decoder(cur_node_emb, cav_features)
        probs = self.policy_head(context, action_mask)
        value = self.value_head(context)

        return probs, value

    def get_encoding(self, state):
        """Pre-compute encoder output for MCTS reuse."""
        B = state.shape[0]
        road_flat = state[:, :self.num_edges * self.road_feature_dim]
        road_features = road_flat.view(B, self.num_edges, self.road_feature_dim)
        return self.encoder(road_features)

    def decode_from_encoding(self, encoding, cav_features, cur_edge_idx=None, action_mask=None):
        """Use pre-computed encoding to get policy and value (for MCTS efficiency)."""
        self.decoder.set_kv(encoding)

        if cur_edge_idx is not None:
            idx = cur_edge_idx.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            cur_node_emb = torch.gather(encoding, 1, idx)
        else:
            cur_node_emb = encoding.mean(dim=1, keepdim=True)

        self.decoder.set_q_first_node(cur_node_emb, cav_features)
        context = self.decoder(cur_node_emb, cav_features)
        probs = self.policy_head(context, action_mask)
        value = self.value_head(context)
        return probs, value
