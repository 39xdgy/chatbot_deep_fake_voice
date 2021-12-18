import re, math
import numpy as np

import nltk
nltk.download('wordnet') 
from nltk.corpus import wordnet

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import torch.nn.functional as F 

from units.MoveData import Options, json2datatools, num_batches, nopeak_mask, create_masks
from models.EncoderDecoder import Encoder, Decoder

def get_synonym(word, field, explain=False):
    syns = wordnet.synsets(word)
    for s in syns:
        if explain: print('synonym:', s.name())
        for l in s.lemmas():
            if explain: print('-lemma:', l.name())
            if field.vocab.stoi[l.name()] != 0:
                if explain: print('found in vocab', l.name())
                return field.vocab.stoi[l.name()]
    return 0 # if we cannot find a synonym, return 0

def string2tensor(string, inputfield, explain=False):
    '''
    input:
        string (str) input sentence
        inputfield a PyTorch torchtext.data.Field object
        explain, set this to True if you want to see how the sentence was split 
    output:
        sequence of tokens (torch tensor of integers) shape  
    '''
    sentence = inputfield.preprocess(string)
    if explain: print(sentence)
    integer_sequence = []
    for tok in sentence:
        if inputfield.vocab.stoi[tok] != 0:
            integer_sequence.append(inputfield.vocab.stoi[tok])
        else:
            integer_sequence.append(get_synonym(tok, inputfield))
    return torch.LongTensor([integer_sequence])

def talk_to_chloe(input_str, model, opt, infield, outfield):
    '''
    input:
        input_str is a string, it is what you want to say to the dialogue model
        model is a Transformer model with encoder, decoder and a last layer linear transformation
        opt is an options object with the maximum length of the output sequence opt.max_len
        infield and outfield are the data.fields that store the vocabulary
    output:
        an output string response from the dialogue model
    
    Note: this version assumes we are evaluating the model on CPU 
    '''
    model.eval()
    model.cpu()
    input_sequence = string2tensor(input_str, infield) # string to tensor 
    input_mask = (input_sequence != infield.vocab.stoi['<pad>']).unsqueeze(-2) #make input mask
    encoding = model.encoder(input_sequence, input_mask) # use the encoder rerepresent the input
    init_tok = outfield.vocab.stoi['<sos>'] # this is the integer for the start token
    decoder_input = torch.LongTensor([[init_tok]]) # use start token to initiate the decoder
    
    # continue obtaining the next decoder token until decoder outputs and end token or til max_len 
    for pos in range(opt.max_len):
        decoder_input_mask = nopeak_mask(size=pos+1, opt=opt) # make target mask, pos+1 casue pos starts at 0
        # the out vector contains the logits that are rebalanced by the softmax
        out = model.out(model.decoder(decoder_input, decoder_input_mask, encoding, input_mask))
        softout = F.softmax(out, dim=-1) 
        #softout is a categorical probability distribution over the output vocab
        distr = Categorical(probs=softout)
        action = distr.sample()[:,-1].unsqueeze(0) # sample from that distribution to get next token
        # concatenate that token to our running list of output tokens 
        decoder_input = torch.cat((decoder_input, action), dim=1) 
        # if the model outputs an end of sentence token, it is done with this sentence
        if outfield.vocab.itos[action] == '<eos>':
            # [0] because we are assuming batch size of 1 
            # [1:-1] excludes the start and end token from the output string 
            de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0][1:-1]])
            return de_str
        
    de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0]])
    return de_str

def talk_to_model(input_str, model, opt, infield, outfield):
    '''
    input:
        input_str is a string, it is what you want to say to the dialogue model
        model is a Transformer model with encoder, decoder and a last layer linear transformation
        opt is an options object with the maximum length of the output sequence opt.max_len
        infield and outfield are the data.fields that store the vocabulary
    output:
        an output string response from the dialogue model
    
    Note: this version assumes we are evaluating the model on CPU 
    '''
    model.eval()
    model.cpu()
    input_sequence = string2tensor(input_str, infield) # string to tensor 
    input_mask = (input_sequence != infield.vocab.stoi['<pad>']).unsqueeze(-2) #make input mask
    #encoding = model.encoder(input_sequence, input_mask, model.memory, model.mem_mask) # use the encoder rerepresent the input
    encoding = model.encoder(input_sequence, input_mask)
    init_tok = outfield.vocab.stoi['<sos>'] # this is the integer for the start token
    decoder_input = torch.LongTensor([[init_tok]]) # use start token to initiate the decoder
    
    # continue obtaining the next decoder token until decoder outputs and end token or til max_len 
    for pos in range(opt.max_len):
        decoder_input_mask = nopeak_mask(size=pos+1, opt=opt) # make target mask, pos+1 casue pos starts at 0
        # the out vector contains the logits that are rebalanced by the softmax
        out = model.out(model.decoder(decoder_input, decoder_input_mask, encoding, input_mask))
        softout = F.softmax(out, dim=-1) 
        #softout is a categorical probability distribution over the output vocab
        distr = Categorical(probs=softout)
        action = distr.sample()[:,-1].unsqueeze(0) # sample from that distribution to get next token
        # concatenate that token to our running list of output tokens 
        decoder_input = torch.cat((decoder_input, action), dim=1) 
        # if the model outputs an end of sentence token, it is done with this sentence
        if outfield.vocab.itos[action] == '<eos>':
            # [0] because we are assuming batch size of 1 
            # [1:-1] excludes the start and end token from the output string 
            de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0][1:-1]])
            return de_str
        
    de_str = ' '.join([outfield.vocab.itos[tok] for tok in decoder_input[0]])
    return de_str