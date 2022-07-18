import math
import torch
import torch.nn as nn
from src.utils.clone import clone_module


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention.
    
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2 -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def generate_subsequent_mask(size):
    """Return mask for output embeddings"""
    attn_shape = (size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        
        # Linear projections of Q, K, V
        q, k, v = [
            l(x).view(n_batches, -1, self.n_head, self.d_k.transpose(1, 2)) 
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # Apply attention on projections
        x, self.attn = attention(q, k, v, mask, self.dropout)
        
        # Concatenate
        x = (x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head ( self.d_k)))
        
        # Free memory
        del query
        del key
        del value
        
        # Apply final linear layer
        return self.linears[-1](x)
