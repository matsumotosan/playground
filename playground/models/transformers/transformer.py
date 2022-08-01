"""PyTorch Lightning re-implementation of a Transformer based on implementations provided
by 'The Annotated Transformer' (http://nlp.seas.harvard.edu/annotated-transformer/) 
and PyTorch (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py).

The resulting re-implementation is akin to a simplified version of the PyTorch implementation.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer"
]


class Transformer(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        n_encoder_layers : int = 6,
        n_decoder_layers : int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        encoder: Union[None, nn.Module] = None,
        decoder: Union[None, nn.Module] = None,
        norm_first: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_model = d_model
        self.n_head = n_head
        
        self.encoder = (self._get_encoder(
            d_model,
            n_head,
            n_encoder_layers,
            dim_feedforward,
            dropout,
            norm_first
        ) if encoder is None else encoder)

        self.decoder = (self._get_decoder(
            d_model,
            n_head,
            n_decoder_layers,
            dim_feedforward,
            dropout,
            norm_first
        ) if decoder is None else decoder)
        
        self.initialize_weights()
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, tgt_mask, memory_mask):
        return self.decoder(tgt, memory, tgt_mask, memory_mask)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        return output
    
    def training_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
    
    def initialize_weights(self, init_range=0.1):
        pass
        # for m in self.encoder.modules():
        #     nn.init.uniform_(m.weight, -init_range, init_range)
        #     nn.init.zeros_(m.bias) 
        # for m in self.decoder.modules():
        #     nn.init.uniform_(m.weight, -init_range, init_range) 
        #     nn.init.zeros_(m.bias) 
    
    def configure_optimizers(self, beta1=0.9, beta2=0.98):
        optimizer = torch.optim.Adam(
            self.parameters(),
            betas=(beta1, beta2)
        )
        scheduler = None
        return [optimizer], [scheduler]

    @staticmethod
    def _get_encoder(d_model, n_head, n_encoder_layers, dim_feedforward, dropout, norm_first):
        encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            norm_first
        )
        encoder = TransformerEncoder(encoder_layer, n_encoder_layers)
        return encoder
    
    @staticmethod
    def _get_decoder(d_model, n_head, n_decoder_layers, dim_feedforward, dropout, norm_first):
        decoder_layer = TransformerDecoderLayer(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            norm_first
        )
        decoder = TransformerDecoder(decoder_layer, n_decoder_layers)
        return decoder
    
    @staticmethod
    def generate_subsequent_mask(size):
        """Return mask for output embeddings"""
        attn_shape = (size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0


class TransformerEncoder(nn.Module):
    """TransformerEncoder consists of a stack of N TransformerEncoderLayer layers."""
    def __init__(self, encoder_layer, n_layers):
        super().__init__()
        self.layers = _clone_layer(encoder_layer, n_layers)
        # self.norm = nn.LayerNorm(encoder_layer.size)
    
    def forward(self, src, src_mask):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask)
        # x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """TransformerDecoder consists of a stack of N TransformerDecoderLayer layers."""
    def __init__(self, decoder_layer, n_layers):
        super().__init__()
        self.layers = _clone_layer(decoder_layer, n_layers)
        # self.norm = nn.LayerNorm(decoder_layer.size)
        
    def forward(self, tgt, memory, tgt_mask, memory_mask):
        x = tgt
        for l in self.layers:
            x = l(x, memory, tgt_mask, memory_mask)
        # x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is composed of a self-attention operation followed by a feed-forward network.
    
    Parameters
    ----------
    d_model : 
    
    n_head : 
    
    dim_feedforward :
    
    dropout : 
    
    norm_first : bool, default=False
        If True, layer norm is performed before self-attention and feed forward operations.
        Otherwise, layer norm is performed after.
    """
    def __init__(self, d_model, n_head, dim_feedforward, dropout, norm_first=False):
        super().__init__()
        
        self.self_attention = SelfAttention(d_model, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, src, mask):
        """Forward pass through an encoder layer. Provides option to perform LayerNorm operation 
        before or after self-attention and feed forward operations."""
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, mask))
            x = self.norm2(x + self._ff_block(x))
            
        return x
    
    def _sa_block(self, x, mask):
        return self.dropout1(self.self_attention(x, x, x, mask))
    
    def _ff_block(self, x):
        return self.dropout2(self.feed_forward(x))


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is composed of a self-attention, multi-head attentionm and feed-forward network."""
    def __init__(self, d_model, n_head, dim_feedforward, dropout, norm_first=False):
        super().__init__()
        self.self_attention = SelfAttention(d_model, n_head)
        self.multihead_attention = SelfAttention(d_model, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Forward pass through a decoder layer. Provides option to perform LayerNorm operation 
        before or after self-attention, multi-head attention, and feed forward operations.
        
        Parameters
        ----------
        tgt : 
            Decoder input
        
        memory :
            Output of last encoder layer
        
        tgt_mask : 
            Mask for decoder input
        
        memory_mask : 
            Mask for outout of last encoder layer
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
            
        return x

    def _sa_block(self, x, attention_mask):
        return self.dropout1(self.self_attention(x, x, x, attention_mask))
    
    def _mha_block(self, x, memory, attention_mask):
        return self.dropout2(self.multihead_attention(x, memory, memory, attention_mask))
    
    def _ff_block(self, x):
        return self.dropout3(self.feed_forward(x))


class SelfAttention(nn.Module):
    """SelfAttention creates a multi-headed self-attention block
    
    Parameters
    ----------
    d_model : int
        Model dimension
    
    n_head : int
        Number of heads
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        assert (d_model % n_head == 0), "Model dimension (d_model) needs to be divisible by number of heads (n_head)"
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.values = nn.Linear(self.d_k, self.d_k, bias=False)
        self.keys = nn.Linear(self.d_k, self.d_k, bias=False)
        self.queries = nn.Linear(self.d_k, self.d_k, bias=False)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, values, keys, queries, mask=None):
        """Performs self-attention operation.
        
        Arguments are ordered as (V, K, Q) as opposed to (Q, K, V) to be consistent with the 
        'Multi-Head Attention' schematic (Fig. 2 (right) from Vaswani et al., 2017).
        
        Parameters
        ----------
        values : Tensor of shape (N, )
        
        keys : Tensor of shape (N, )
        
        queries : Tensor of shape (N, q, )
        
        mask : Tensor of shape () 
            Attention mask, default=None
        """
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


def _clone_layer(layer, N):
    """Returns a ModuleList of N identical modules."""
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
