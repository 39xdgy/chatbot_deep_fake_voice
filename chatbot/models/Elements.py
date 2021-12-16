import time, sys, math, copy
import numpy as np
from matplotlib import pyplot as plt 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from units.MoveData import Options, json2datatools, num_batches, nopeak_mask, create_masks

class Transformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.encoder = Encoder(in_vocab_size, emb_dim, n_layers, heads, dropout)
        self.decoder = Decoder(out_vocab_size, emb_dim, n_layers, heads, dropout)
        self.out = nn.Linear(emb_dim, out_vocab_size)
    def forward(self, src_seq, src_mask, trg_seq,  trg_mask):
        e_output = self.encoder(src_seq, src_mask)
        d_output = self.decoder(trg_seq, trg_mask, e_output, src_mask)
        output = self.out(d_output)
        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, emb_dim)
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
    def forward(self, src_seq, mask):
        x = self.embed(src_seq)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x

class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):

    def __init__(self, emb_dim, max_seq_len = 3000, dropout = 0.1):
        super().__init__()
        self. emb_dim =  emb_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        # this matrix of shape (1, input_seq_len, emb_dim) is
        # cut down to shape (1, input_seq_len, emb_dim) in the forward pass
        # to be broadcasted across each sample in the batch 
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                wavelength = 10000 ** ((2 * i)/ emb_dim)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0) #add a batch dimention to your pe matrix 
        self.register_buffer('pe', pe) #block of data(persistent buffer)->temporary memory
 
    def forward(self, x):
        '''
        input: sequence of vectors  
               shape (batch size, input sequence length, vector dimentions)
        output: sequence of vectors of same shape as input with positional
                aka time encoding added to each sample 
                shape (batch size, input sequence length, vector dimentions)
        '''
        x = x * math.sqrt(self.emb_dim) # make embeddings relatively larger
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe #add constant to embedding
        return self.dropout(x)

def get_clones(module, N):
    '''
    example usage:
        
        # initialize n_layers deep copies of the same encoder
        self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        
        # usage, apply n_layers transformations to x
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)  
    '''
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
    
    def forward(self, x, explain=False):
        x = self.dropout(F.leaky_relu(self.linear_1(x)))
        if explain: print('hidden layer output',x)
        x = self.linear_2(x)
        return x

class repeatFeedForward(nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats
        self.layers = get_clones(FeedForward(emb_dim=4, ff_dim=5, dropout=0.2), repeats)
    def forward(self, x, explain=True):
        for i in range(self.repeats):
            x = self.layers[i](x, explain=explain)
        return x

class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.size = emb_dim
        # alpha and bias are learnable parameters that scale and shift
        # the representations respectively, aka stretch and translate 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps #prevents divide by zero explosions 
    
    def forward(self, x):
        '''
        input: x, shape (batch size, sequence length, embedding dimensions)
        output: norm, shape (batch size, sequence length, embedding dimensions)
        '''
        norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        norm = self.alpha * norm + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, dim_k = None, dropout = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.dim_k = dim_k if dim_k else emb_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.k_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.v_linear = nn.Linear(emb_dim,self.dim_k*num_heads)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim_k*num_heads,emb_dim)
    
    def attention(self, q, k, v, dim_k, mask=None, dropout=None, explain=False):
        k = k.transpose(-2, -1)
        if explain: print('q, k', q.shape, k.shape)
        # matrix multiplication is done using the last two dimensions
        # (batch_size,num_heads,q_seq_len,dim_k)X(batch_size,num_heads,dim_k,k_seq_len)
        # = (batch_size,num_heads,q_seq_len,k_seq_len)
        scores = torch.matmul(q, k) / math.sqrt(dim_k) 
        if explain: print('scores.shape', scores.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            if explain: print('mask.shape', mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9) 
        softscores = F.softmax(scores, dim=-1)
        if dropout is not None: softscores = dropout(softscores)
            
        #(batch_size,num_heads,q_seq_len, k_seq_len)X(batch_size,num_heads,v_seq_len,dim_v)
        output = torch.matmul(softscores, v)
        return output, scores #=(batch_size,num_heads,q_seq_len,dim_v)
    
    def forward(self, q, k, v, mask=None, explain=False):
        '''
        inputs:
            q has shape (batch size, q_sequence length, embedding dimensions)
            k,v are shape (batch size, kv_sequence length, embedding dimensions)
            source_mask of shape (batch size, 1, kv_sequence length)
        outputs: sequence of vectors, re-represented using attention
            shape (batch size, q_sequence length, embedding dimensions)
        use:
            The encoder layer places the same source vector sequence into q,k,v 
            and source_mask into mask.
            The decoder layer uses this twice, once with decoder inputs as q,k,v 
            and target mask as mask. then with decoder inputs as q, encoder outputs
            as k, v and source mask as mask
        '''
        # k,q,v are each shape (batch size, sequence length, dim_k * num_heads)
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        if explain: print("(batch size, sequence length, dim_k * num_heads)", k.shape)
        # k,q,v are each shape (batch size, sequence length, num_heads, dim_k)
        k = k.view(batch_size,-1,self.num_heads,self.dim_k)
        q = q.view(batch_size,-1,self.num_heads,self.dim_k)
        v = v.view(batch_size,-1,self.num_heads,self.dim_k)
        # transpose to shape (batch_size, num_heads, sequence length, dim_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        if explain: print("(batch_size,num_heads,seq_length,dim_k)",k.shape)
        # calculate attention using function we will define next
        attn, scores = self.attention(q, k, v, self.dim_k, mask, self.dropout, explain)
        if explain: print("attn(batch_size,num_heads,seq_length,dim_k)", attn.shape)
        # concatenate heads, dim_k = dim_v
        concat=attn.transpose(1,2).contiguous().view(batch_size,-1,self.dim_k*self.num_heads)
        if explain: print("concat.shape", concat.shape)
        # put through final linear layer
        output = self.out(concat)
        if explain: print("MultiHeadAttention output.shape", output.shape)
        return output, scores