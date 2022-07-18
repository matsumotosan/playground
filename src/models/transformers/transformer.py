"""PyTorch Lightning re-implementation of a Transformer based largely on implementation provided
by 'The Annotated Transformer'.
http://nlp.seas.harvard.edu/annotated-transformer/
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from types import Union
from ...utils.clone import clone_module


class Transformer(pl.LightningModule):
    def __init__(
        self,
        encoder: Union[None, nn.Module] = None,
        decoder: Union[None, nn.Module] = None,
        generator: Union[None, nn.Module] = None,
        n_encoder_layers : int = 6,
        n_decoder_layers : int = 6,
        d_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 2048
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_model = d_model
        self.n_head = n_head
        self.encoder = encoder
        self.decoder = decoder
        # self.src_embed = src_embed
        # self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        mem = self.encode(src, src_mask)
        return self.decode(mem, src_mask, tgt, tgt_mask)
    
    def training_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
    
    def configure_optimizers(self):
        pass


class SublayerConnection(nn.Module):
    """Residual connection followed by a layer norm.
    Modular building block for encoder and decoder stacks.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    """Encoder class"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone_module(layer, N)
        self.layer_norm = nn.LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    """Single encoder layer"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_module(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Decoder class"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone_module(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
    
class DecoderLayer(nn.Module):
    """Single decoder layer"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_module(SublayerConnection(size, dropout), 3)
        
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
