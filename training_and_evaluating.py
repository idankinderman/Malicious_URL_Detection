# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from preprocessing import BuildDataLoaderNoTokenizatie


def Evaluate(model, data_loader):
    data_iterator = iter(data_loader)
    nb_batches = len(data_loader)
    model.eval()
    acc = 0 
    for url, label in data_iterator:
        out = model(url)
        acc += (out.argmax(1) == label).cpu().numpy().mean()

    print(f"eval accuracy: {acc / nb_batches}")
    return acc / nb_batches


def Train(model, train_loader, valid_loader, test_loader, loss_func, optimizer, epochs):
    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    for epoch in range(epochs):
        train_iterator = iter(train_loader)
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for urls, label in train_iterator:
            out = model(urls)
            loss = loss_func(out, label)
            model.zero_grad()

            loss.backward()
            losses += loss.item()

            optimizer.step()
                        
            train_acc += (out.argmax(1) == label).cpu().numpy().mean()
            
        train_loss.append(losses / nb_batches_train)
        train_accuracy.append(train_acc / nb_batches_train)
        print("\n","-"*10 ,"epoch number", epoch, "-"*10, "\n")
        print(f"train loss: {losses / nb_batches_train}")
        print(f"training accuracy: {train_acc / nb_batches_train}")
        print("accuracy of validation set:")
        validation_accuracy.append(Evaluate(model, valid_loader))
    
    print()
    print("Done training")
    print("accuracy of test set:")
    Evaluate(model, test_loader)
    return train_loss, train_accuracy, validation_accuracy

def DrawGraphs(train_loss, train_accuracy, validation_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(train_accuracy, label="train set")
    ax.plot(validation_accuracy, label="validation set")
    ax.set_xlabel("epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax = fig.add_subplot(212)
    ax.plot(train_loss)
    ax.set_xlabel("epochs")
    ax.set_ylabel("Training loss")


