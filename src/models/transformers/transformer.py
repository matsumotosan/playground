"""PyTorch Lightning re-implementation of a Transformer based on implementations provided
by 'The Annotated Transformer'.
http://nlp.seas.harvard.edu/annotated-transformer/
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from types import Union


class Transformer(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        encoder: Union[None, nn.Module] = None,
        decoder: Union[None, nn.Module] = None,
        generator: Union[None, nn.Module] = None,
        n_encoder_layers : int = 6,
        n_decoder_layers : int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_model = d_model
        self.n_head = n_head
        
        if encoder is not None:
            self.encoder = encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
            self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers)
            
        if decoder is not None:
            self.decoder = decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout)
            self.decoder = TransformerDecoder(decoder_layer, n_decoder_layers)
            
        # self.src_embed = src_embed
        # self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
    
    def training_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
    
    def configure_optimizers(self, beta1=0.9, beta2=0.98):
        optimizer = torch.optim.Adam(
            self.parameters(),
            betas=(beta1, beta2)
        )
        scheduler = None
        return [optimizer], [scheduler]


class TransformerEncoder(nn.Module):
    """Encoder stack"""
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = _clone_layer(layer, n_layers)
        self.layer_norm = nn.LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class TransformerDecoder(nn.Module):
    """Decoder stack"""
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = _clone_layer(layer, n_layers)
        self.layer_norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask)
        return self.layer_norm(x)


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is composed of a self-attention followed by a feed forward network."""
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        
        self.self_attention = SelfAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x

    
class TransformerDecoderLayer(nn.Module):
    """Single decoder layer"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _clone_layer(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x
    
    
class Generator(nn.Module):
    """Linear layer followed by softmax."""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SelfAttention(nn.Module):
    """Mult-head attention class
    
    Parameters
    ----------
    d_model : int, default=512

    
    n_head : int, default=8
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        assert (d_model % n_head == 0), "Model dimension (d_model) needs to be divisible by number of heads (n_head)"
        self.d_k = d_model // n_head
        self.n_head = n_head
        
        self.values = nn.Linear(self.d_k, self.d_k, bias=False)
        self.keys = nn.Linear(self.d_k, self.d_k, bias=False)
        self.queries = nn.Linear(self.d_k, self.d_k, bias=False)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, values, keys, queries, mask=None):
        # Get dimensions of values, keys, and queries
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # Split embeddings into self.n_head pieces
        values = values.reshape(N, value_len, self.n_head, self.d_k)
        keys = keys.reshape(N, key_len, self.n_head, self.d_k)
        queries = queries.reshape(N, query_len, self.n_head, self.d_k)
        
        # Matrix multiply queries and keys
        # queries shape: (N, query_len, n_head, d_k)    -- (n, q, h, d)
        # keys shape: (N, key_len, n_head, d_k)         -- (n, k, h, d)
        # energy shape: (N, n_head, query_len, key_len) -- (n, h, q, k)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        
        # Mask attention weights
        if mask is not None:
            energy.masked_fill_(mask == 0, float("-1e20"))
        
        # Calculate proportion of attention for each key for each query
        p_attention = F.softmax(energy / math.sqrt(self.d_k), dim=3)
        
        # Calculate attention and concatenate (collapse last two dimensions)
        # p_attention shape: (N, n_head, query_len, key_len)
        # values shape: (N, value_len, n_head, d_k)
        # attention shape: (N, query_len, n_head, d_k)
        # Note: key_len == value_len
        attention = torch.einsum("nhql,nlhd->nqhd", p_attention, values).reshape(
            N, query_len, self.n_head * self.d_k
        )
        
        # Apply final linear layer
        return self.fc_out(attention)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.aragnge(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# class Sublayer(nn.Module):
#     """Modular sublayer class for building encoder and decoder stacks.
#     Output of each sublayer is LayerNorm(x + Sublayer(x)) where the Sublayer is a
#     multi-head attention layer or a feed forward layer.
    
#     Parameters
#     ----------
#     size : int
#         Output dimension of sublayer

#     dropout : float
#         Dropout probability
#     """
#     def __init__(self, size, dropout):
#         super().__init__()
#         self.norm = nn.LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))
#         # return self.norm(x + sublayer(x))

def _clone_layer(layer, N):
    """Returns a ModuleList of N identical modules."""
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

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
