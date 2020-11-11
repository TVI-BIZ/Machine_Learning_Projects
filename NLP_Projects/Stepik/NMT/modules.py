import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x): # с tempreture=10, отвечает за гладкость
    e_x = torch.exp(x / 10)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim] - H from encoder
        # hidden = [1, batch size, dec_hid_dim] - st-1
        
        # repeat hidden and concatenate it with encoder_outputs
        repeat = hidden.repeat(encoder_outputs.shape[0],1,1)
        summation = torch.cat((repeat,encoder_outputs),2)
        
        # calculate energy
        energy = self.attn(summation).tanh() #'''your code'''        
        w_vector = torch.matmul(energy,hidden.permute(0,2,1))
    
        # get attention, use softmax function which is defined, can change temperature
        #'''your code'''
        attn = softmax(self.v(energy))
            
        return attn #'''your code'''
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim,emb_dim) #'''your code'''
        
        self.rnn = nn.GRU(dec_hid_dim,dec_hid_dim,dropout=dropout) #'''your code''' # use GRU
        
        self.out = nn.Linear(dec_hid_dim,output_dim)  #'''your code''' # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):     
        input = input.unsqueeze(0) # because only one word, no words sequence 
        #print(input.shape,'input_shape_after_unsq')

        embedded = self.dropout(self.embedding(input))

        # get weighted sum of encoder_outputs
        
        attn_vector = self.attention(hidden,encoder_outputs)
        weighted_sum = torch.bmm(encoder_outputs.permute(1,2,0),attn_vector.permute(1,0,2))
       
        
        
        
        
        # concatenate weighted sum and embedded, break through the GRU
        conc_weight = weighted_sum.permute(2,0,1)+embedded
        rnn_out,rnn_hidden = self.rnn(conc_weight,hidden)    
        three_conc = embedded + rnn_out + weighted_sum.permute(2,0,1)
        
        # get predictions
        preds = self.out(three_conc)
        preds = preds.squeeze(0)

        return preds,rnn_hidden
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            #print(output.shape,'output_shape_seq2seq')
            
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
