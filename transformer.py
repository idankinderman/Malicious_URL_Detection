# -*- coding: utf-8 -*-
# imports for the tutorial
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
from preprocessing import BuildDataLoaderNoTokenizatie
from training_and_evaluating import Train, DrawGraphs
from embedding_and_positional_encoding import Embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
            
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads
        
        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0) 
        k_length = K.size(-2) 
        
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        
        A = torch.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        A = self.dropout(A)
        
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)

        return H, A 

        
    def split_heads(self, x, batch_size):
        
        #Split the last dimension into (heads X depth)
        #Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
    
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        
        #Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads 
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)
        
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)
        
        # Final linear layer  
        H = self.W_h(H_cat)          # (bs, q_length, dim)
        
        return H, A


#1-D (1x1) Convolution: an MLP with one hidden layer and ReLU activation applied to each and every element in the set.

class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model, hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cnn = CNN(d_model, conv_hidden_dim)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
    
    def forward(self, x):
        
        # Multi-head attention 
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        # Feed forward 
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)
        
        # Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = Embeddings(d_model, dropout, with_positional=True)

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, conv_hidden_dim, self.dropout))
        
    def forward(self, x):
        x = self.embedding(x) # Transform to (batch_size, input_seq_length, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)
  
# Transormer classifier for sentiment analysis
class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, num_answers=2, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, conv_hidden_dim, dropout)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dense = nn.Linear(d_model, num_answers)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        x = self.softmax(x) 
        return x

train_dataloader, val_dataloader, test_dataloader = BuildDataLoaderNoTokenizatie(url_num=120_000, batch_size=16, num_classes=2)
model = TransformerClassifier(num_layers=2, d_model=32, num_heads=16, conv_hidden_dim=128, num_answers=2)
loss = f.cross_entropy
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("The model have", pytorch_total_params, "parameters")
start = time.time()
train_loss, train_acc, validation_acc, validation_acc_focus = Train(model, train_dataloader, val_dataloader, test_dataloader, loss,
                                              optimizer, epochs=17, focus_start=100, focus_end=99)
end = time.time()
print("The training took", '{:.6}'.format(end - start) ,"seconds")
DrawGraphs(train_loss, train_acc, validation_acc, validation_acc_focus, "Transformer")



