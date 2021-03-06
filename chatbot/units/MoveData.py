import re
import numpy as np
from nltk.tokenize import TweetTokenizer

import torch
from torchtext.legacy import data
from torch.autograd import Variable

class Options:
    def __init__(self, batchsize=4, device=-1, epochs=20, lr=0.01, 
                 beam_width=2, max_len=20, 
                 save_path='/home/wyundi/Server/Courses/BIA667/Project/Chatbot/saved/weights/model_weights'):
        self.batchsize = batchsize
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.k = beam_width
        self.max_len = max_len
        self.save_path = save_path

class MyIterator(data.Iterator):
    '''
    patch on Torchtext's batching process that makes it more efficient
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
    '''
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
                
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class Tokenizer(object):
    
    def __init__(self):

        self.tweettokenizer = TweetTokenizer()
            
    def tokenize(self, sentence):
        sentence = re.sub(r"[\*\"??????\n\\???\+\-\/\=\(\)??????:\[\]\|???\;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        sentence = self.tweettokenizer.tokenize(sentence)
        return sentence
    
def num_batches(train):

    for i, b in enumerate(train):
        pass
    
    return i + 1

def json2datatools(path = None, tokenizer = None, opt = None):

    if opt == None:
        opt = Options()
        opt.batchsize = 4
        opt.device = torch.device("cpu")

    if path == None:
        path = 'saved/pairs.json' 

    if tokenizer == None:
        tokenizer = Tokenizer()
        
    input_field = data.Field(lower=True, tokenize=tokenizer.tokenize)
    output_field = data.Field(lower=True, tokenize=tokenizer.tokenize, 
                              unk_token='<unk>', init_token='<sos>', eos_token='<eos>')

    fields={'listen':('listen', input_field),'reply':('reply', output_field)} 

    trainingset = data.TabularDataset(path, format='json', fields=fields) 

    input_field.build_vocab(trainingset)
    output_field.build_vocab(trainingset)
    training_iterator = MyIterator(trainingset, batch_size=opt.batchsize, 
                        device=torch.device('cpu'), repeat=False, 
                        sort_key=lambda x: (len(x.listen), len(x.reply)), 
                        train=True, shuffle=True)
    opt.src_pad = input_field.vocab.stoi['<pad>']
    opt.trg_pad = output_field.vocab.stoi['<pad>']
    return training_iterator, input_field, output_field, opt

def nopeak_mask(size, opt):
    "Mask out subsequent positions. aka subsequent_mask"
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.from_numpy(np_mask) == 0
    return np_mask

def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def load_subset_weights(whole_model, opt):
    '''
    This function allows you to load saved weights from a saved model that is a subset of your model
    It looks for the named parameters that match and loads those but will not crash trying to load
    parameters that dont have a matching name
    '''
    subset_model_dict = torch.load(opt.save_path)
    whole_model_dict = whole_model.state_dict() 
    for name, param in whole_model_dict.items(): 
        if name in subset_model_dict:
            whole_model_dict[name].copy_(subset_model_dict[name])