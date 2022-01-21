# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from preprocessing import accepted_chars


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        X = X / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :] # for batch first
        return self.dropout(X)
    
    
# Embeddings class: sequences -> features
class Embeddings(nn.Module):
    def __init__(self, d_model, dropout=0, with_positional=False):
        super().__init__()
        self.dropout = dropout
        self.vocab = ['<pad>'] + list(accepted_chars) + ['<SOS>', '<EOS>']
        self.word_embeddings = nn.Embedding(len(self.vocab), d_model, padding_idx=1)
        self.position_embeddings = PositionalEncoding(num_hiddens=d_model, dropout=self.dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.d_model = d_model
        self.with_positional = with_positional

    def pre_embedding(self, seqs):
        vectorized_seqs = [[self.vocab.index(tok) for tok in seq]for seq in seqs]
        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max()+2)).long()

        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx,0] = len(self.vocab) - 2 # adding the char '<SOS>'
            seq_tensor[idx, 1:seqlen+1] = torch.LongTensor(seq)
            seq_tensor[idx,seqlen+1] = len(self.vocab) - 1 # adding the char '<EOS>'
        
        return seq_tensor

    def forward(self, batch):
        batch = self.pre_embedding(batch)
        # Get word embeddings for each input id
        word_embeddings = self.word_embeddings(batch)                   # (bs, max_seq_length, dim)
        # Get position embeddings for the word embeddings and add them  
        if self.with_positional:
            word_embeddings = self.position_embeddings(word_embeddings) # (bs, max_seq_length, dim)
        # Layer norm 
        embeddings = self.LayerNorm(word_embeddings)             # (bs, max_seq_length, dim)
        return embeddings