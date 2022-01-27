# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from preprocessing import BuildDataLoaderNoTokenizatie


def Evaluate(model, data_loader, cheap_mode=False):
    data_iterator = iter(data_loader)
    nb_batches = len(data_loader)
    nb_batches = len(data_loader)/12 if cheap_mode else len(data_loader)
    model.eval()
    acc = 0 
    
    for count, (url, label) in enumerate(data_iterator):
        out = model(url)
        acc += (out.argmax(1) == label).cpu().numpy().mean()
        if cheap_mode and count >= nb_batches:
            break
        
    if not cheap_mode:
        print(f"eval accuracy: {acc / nb_batches}")
    return acc / nb_batches


def Train(model, train_loader, valid_loader, test_loader, loss_func, optimizer,
          epochs, focus_start, focus_end):
    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    validation_accuracy_focus = []
    for epoch in range(epochs):
        train_iterator = iter(train_loader)
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for count, (urls, label) in enumerate(train_iterator):
            out = model(urls)
            loss = loss_func(out, label)
            model.zero_grad()

            loss.backward()
            losses += loss.item()
            optimizer.step()
            train_acc += (out.argmax(1) == label).cpu().numpy().mean()
            
            if (epoch >= focus_start and epoch <= focus_end and count % 4 == 0 and count<=nb_batches_train/4):
                validation_accuracy_focus.append(Evaluate(model, valid_loader, cheap_mode=True))
                
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
    return train_loss, train_accuracy, validation_accuracy, validation_accuracy_focus

def DrawGraphs(train_loss, train_accuracy, validation_accuracy, validation_acc_focus, model_name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(train_accuracy, label="train set")
    ax1.plot(validation_accuracy, label="validation set")
    ax1.set_title("Accuracy on {}".format(model_name))
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(train_loss)
    ax2.set_title("Training loss on {}".format(model_name))
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Loss")
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.plot(validation_acc_focus)
    ax3.set_title("Accuracy on validation - focused epochs - on {}".format(model_name))
    ax3.set_xlabel("4 batches")
    ax3.set_ylabel("Accuracy")
    