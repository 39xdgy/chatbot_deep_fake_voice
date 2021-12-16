import math, copy, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.Elements import *

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.norm_2 = Norm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, vector_sequence, mask):
        '''
        input:
            vector_sequence of shape (batch size, sequence length, embedding dimensions)
            source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: sequence of vectors after embedding, postional encoding, attention and normalization
            shape (batch size, sequence length, embedding dimensions)
        '''
        x2 = self.norm_1(vector_sequence)
        x2_attn, x2_scores = self.attn(x2,x2,x2,mask)
        vector_sequence = vector_sequence + self.dropout_1(x2_attn)
        x2 = self.norm_2(vector_sequence)
        vector_sequence = vector_sequence + self.dropout_2(self.ff(x2))
        return vector_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, source_sequence, source_mask):
        '''
        input:
            source_sequence (sequence of source tokens) of shape (batch size, sequence length)
            source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: sequence of vectors after embedding, postional encoding, attention and normalization
            shape (batch size, sequence length, embedding dimensions)
        '''
        vector_sequence = self.embed(source_sequence)
        vector_sequence = self.pe(vector_sequence)
        for i in range(self.n_layers):
            vector_sequence = self.layers[i](vector_sequence, source_mask)
        vector_sequence = self.norm(vector_sequence)
        return vector_sequence

class DecoderLayer(nn.Module):

    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)

    def forward(self, de_out, de_mask, en_out, en_mask):
        '''
        inputs:
            de_out - decoder ouputs so far (batch size, output sequence length, embedding dimensions)
            de_mask (batch size, output sequence length, output sequence length)
            en_out - encoder output (batch size, input sequence length, embedding dimensions)
            en_mask (batch size, 1, input sequence length)
        ouputs:
            de_out (next decoder output) (batch size, output sequence length, embedding dimensions)
        '''
        de_nrm = self.norm_1(de_out)
        #Self Attention 
        self_attn, self_scores = self.attn_1(de_nrm, de_nrm, de_nrm, de_mask)
        de_out = de_out + self.dropout_1(self_attn)
        de_nrm = self.norm_2(de_out)
        #DecoderEncoder Attention
        en_attn, en_scores = self.attn_2(de_nrm, en_out, en_out, en_mask) 
        de_out = de_out + self.dropout_2(en_attn)
        de_nrm = self.norm_3(de_out)
        de_out = de_out + self.dropout_3(self.ff(de_nrm))
        return de_out

class Decoder(nn.Module):
    '''
    If your target sequence is `see` `ya` and you want to train on the entire 
    sequence against the target, you would use `<sos>` `see`  `ya`
    as the de_out (decoder ouputs so far) and compare the 
    output de_out (next decoder output) `see` `ya` `<eos>` 
    as the target in the loss function. The inclusion of the `<sos>`
    for the (decoder ouputs so far) and `<eos>` for the 
    '''
    def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(DecoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, de_toks, de_mask, en_vecs, en_mask):
        '''
        inputs:
            de_toks - decoder ouputs so far (batch size, output sequence length)
            de_mask (batch size, output sequence length, output sequence length)
            en_vecs - encoder output (batch size, input sequence length, embedding dimensions)
            en_mask (batch size, 1, input sequence length)
        outputs:
            de_vecs - next decoder output (batch size, output sequence length, embedding dimensions)

        '''
        x = self.embed(de_toks)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, de_mask, en_vecs, en_mask)
        return self.norm(x)

