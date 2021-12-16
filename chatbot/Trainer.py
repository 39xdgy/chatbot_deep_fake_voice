import time, sys, os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from units.Talk import *
from units.MoveData import Options, json2datatools, num_batches, nopeak_mask, create_masks
from models.EncoderDecoder import Encoder, Decoder

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparing the Data
data_path = '/home/wyundi/Server/Courses/BIA667/Project/data/Kaggle/Mental_Health_FAQ.json'
save_path = '/home/wyundi/Server/Courses/BIA667/Project/Chatbot/saved/models/mental_health_12_12'
opt = Options(batchsize=2, device=device, epochs=300, 
                  lr=0.01, beam_width=3, max_len = 25, save_path = save_path)

data_iter, infield, outfield, opt = json2datatools(path=data_path, opt=opt)

print('input vocab size', len(infield.vocab), 'output vocab size', len(outfield.vocab))

# model
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

emb_dim, n_layers, heads, dropout = 300, 2, 8, 0.01 
model = Transformer(len(infield.vocab), len(outfield.vocab), emb_dim, n_layers, heads, dropout)
# model.load_state_dict(torch.load(opt.save_path))
    
print(model)

def trainer(model, data_iterator, options):

    model = model.to(options.device)

    model.train()
    start = time.time()
    best_loss = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    total_num = num_batches(data_iterator)+1

    for epoch in range(options.epochs):
        total_loss = 0
        for i, batch in enumerate(data_iterator):

            src = batch.listen.transpose(0,1) # get an input ready
            trg = batch.reply.transpose(0,1)  # get a target ready

            trg_input = trg[:, :-1] # trg_input is exactly 1 timestep behind target (ys)
            src_mask, trg_mask = create_masks(src, trg_input, options) # get the mask ready

            src = src.to(options.device)
            trg = trg.to(options.device)
            trg_input = trg_input.to(options.device)
            src_mask = src_mask.to(options.device)
            trg_mask = trg_mask.to(options.device)

            preds = model(src, src_mask, trg_input, trg_mask)

            ys = trg[:, 1:].contiguous().view(-1) # target. 
            
            batch_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), 
                                         ys, ignore_index = options.trg_pad)

            optimizer.zero_grad() # clean out the gradients
            batch_loss.backward() # calculate the gradient
            optimizer.step() # update the weights

            total_loss += batch_loss.item()

        epoch_loss = total_loss / total_num
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), options.save_path)
        print("%dm: epoch %d loss = %.3f" %((time.time() - start)//60, epoch+1, epoch_loss))
        total_loss = 0

    return model

model = trainer(model, data_iter, opt)

if os.path.exists(opt.save_path):
    os.remove(opt.save_path)

print(opt.save_path)
torch.save(model.state_dict(), opt.save_path)


tell_model = "hi i am vicki" 
model_reply = talk_to_model(tell_model, model, opt, infield, outfield)
print('bot > '+ model_reply + '\n')

while True:
    tell_model = input("You > ")
    model_reply = talk_to_model(tell_model, model, opt, infield, outfield)

    if model_reply[:5] == '<sos>':
        model_reply = model_reply[5:]

    if ("bye" in tell_model or "bye ttyl" in model_reply):
        print('bot > '+ model_reply + '\n')
        break
    else:
        print('bot > '+ model_reply + '\n') 
