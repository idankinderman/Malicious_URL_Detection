# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
from preprocessing import BuildDataLoaderNoTokenizatie 
from training_and_evaluating import Train, DrawGraphs
from embedding_and_positional_encoding import Embeddings 


class LSTM_classifier(nn.Module):
    def __init__(self, d_model, hidd_size, lstm_depth, num_answers):
        super().__init__()
        self.embeddings = Embeddings(d_model, dropout=0.2, with_positional=False)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidd_size, num_layers=lstm_depth, batch_first=True)
        self.classifier = nn.Linear(hidd_size, num_answers)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.embeddings(x)
        x, (h,c) = self.lstm(x)
        #x = x[:,-1,:] #giving the last hidden state # no, this is the output!
        x = h[-1,:,:] #giving the last hidden state
        x = self.classifier(x)
        x = self.softmax(x)
        return x
        

train_dataloader, val_dataloader, test_dataloader = BuildDataLoaderNoTokenizatie(url_num=80_000, batch_size=128, num_classes=2)
model = LSTM_classifier(d_model=8, hidd_size=32, lstm_depth=2, num_answers=2)
pytorch_total_params = sum(p.numel() for p in model.parameters())
loss = f.cross_entropy
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print("The model have", pytorch_total_params, "parameters")
start = time.time()
train_loss, train_acc, validation_acc = Train(model, train_dataloader, val_dataloader, test_dataloader, loss, optimizer, epochs=13)
end = time.time()
print("The training took", '{:.6}'.format(end - start) ,"seconds")
DrawGraphs(train_loss, train_acc, validation_acc)
print("\n\n\ntesting mode:")

train_dataloader, val_dataloader, test_dataloader = BuildDataLoaderNoTokenizatie(url_num=20, batch_size=1, num_classes=2)
train_iterator = iter(train_dataloader)
i=1
for urls, label in train_iterator:
    print("\niteration number", i)
    print(label) 
    print(model(urls))
    i = i + 1