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
        

train_dataloader, val_dataloader, test_dataloader = BuildDataLoaderNoTokenizatie(url_num=60_000, batch_size=128, num_classes=2)
model = LSTM_classifier(d_model=8, hidd_size=32, lstm_depth=2, num_answers=2)
pytorch_total_params = sum(p.numel() for p in model.parameters())
loss = f.cross_entropy
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print("The model have", pytorch_total_params, "parameters")
start = time.time()
train_loss, train_acc, validation_acc, validation_acc_focus = Train(model, train_dataloader, val_dataloader, 
                                                                    test_dataloader, loss, optimizer, epochs=24, 
                                                                    focus_start=25, focus_end=24)
end = time.time()
print("\nThe training took", '{:.6}'.format(end - start) ,"seconds")
DrawGraphs(train_loss, train_acc, validation_acc, validation_acc_focus, model_name="LSTM")
print("\n\n\ntesting mode:")

