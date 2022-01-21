# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from collections import Counter
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


# global variables
accepted_chars = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
accepted_chars_after_split = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,:;<>?@[\\]^`{|}~'
splits_chars = ['://', '//', '/', '.', '_', '=', '-'] # we will tokanize according to those
n_letters = len(accepted_chars_after_split)


class URLDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls = urls
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.urls[idx], self.labels[idx]


# convert upper chars to lower chars, and deletes damaged urls
def CharPreprocessing(urls, labels):
    urls = [url.lower() for url in urls]
    bad_url_chars = [i for i, url in enumerate(urls) if bool(set(url).difference(accepted_chars))] # deleting url if its have a unrecognized char
    urls = np.delete(urls, bad_url_chars)
    labels = np.delete(labels, bad_url_chars)
    #badamount = sum ( labels != 'benign')
    return urls,labels 


# get size of above arrays and permute
# returns only N elements from the array
def SampleData(urls,labels,N): 
    urls_num = urls.shape[0]
    idxs=np.random.permutation(urls_num)
    urls = urls[idxs]
    labels = labels[idxs]
    #get first N
    labels = labels[:N]
    urls = urls[:N]
    return urls,labels


# if all_url = false, sampling #url_num urls, half benign and half isn't
# if all_url = true, return all the urls
def UrlReader(url_num, all_url, num_classes=2):
    # reading the database
    df = pd.read_csv('malicious_phish_CSV.csv')
    urls = df.iloc[:,0]
    labels = df.iloc[:,1]
    labels = labels.to_numpy()
    urls = urls.to_numpy()
    
    urls, labels = CharPreprocessing(urls, labels)
    
    if all_url:
        return urls, labels
    
    # takes benign urls and labels
    benign_urls = urls[labels=='benign']
    benign_labels = labels[labels=='benign']
    benign_urls, benign_labels = SampleData(benign_urls, benign_labels, math.floor(url_num/2));
    
    # takes not benign urls and labels
    malware_urls = urls[labels!='benign']
    malware_labels = labels[labels!='benign']
    malware_urls , malware_labels = SampleData(malware_urls, malware_labels, url_num - math.floor(url_num/2));
    
    # creates the urls dataset
    urls = np.concatenate((malware_urls,benign_urls))
    labels = np.concatenate((malware_labels,benign_labels))
    if (num_classes == 2): #converting to binary problem, 0 is 'benign' and 1 is 'malware'
        labels = torch.tensor([1 if label!='benign' else 0 for label in labels]) 
    if (num_classes == 4): #0 is 'benign', 1 is 'malware', 2 is 'phishing', 3 is 'defacement'
        labels = torch.tensor([LabelToInt(label) for label in labels]) 
    return urls, labels

def LabelToInt(label):
    switcher = {
        "benign" : 0,
        "malware" : 1,
        "phishing" : 2,
        "defacement" : 3,
    }
    return switcher.get(label, 4)

# tokanize the url according to a list of chars
def UrlTokenization(seqs):
    for i in range(seqs.shape[0]):    
        for char in splits_chars:
            seqs[i] = np.char.replace(seqs[i], char, " ")
    seqs = [str(url).split() for url in seqs]
    return seqs


# builds DataLoaders of urls without Tokenization
def BuildDataLoaderNoTokenizatie(url_num, batch_size, num_classes=2):
    urls, labels = UrlReader(url_num, False, num_classes)
    # splits to train, validation and test sets
    train_set, test_set, train_labels, test_labels = train_test_split(urls, labels, test_size=0.4, random_state=42)
    val_set, test_set, val_labels, test_labels = train_test_split(test_set, test_labels, test_size=0.5, random_state=42)
    
    dataset_train = URLDataset(train_set, train_labels)
    dataset_val = URLDataset(val_set, val_labels)
    dataset_test = URLDataset(test_set, test_labels)
    
    # create data loaders
    train_dataloader = DataLoader(dataset_train, batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader


# calculates hat is the portion of the num_words most common words, from all the urls words
def WordsStatistics(num_words):
    urls, _ = UrlReader(0, True)
    urls_tokenize = UrlTokenization(urls)
    list_of_words = []
    for url in urls_tokenize:
        for word in url:
            list_of_words.append(word)
    num_of_words = len(list_of_words)
    my_counter = Counter(list_of_words)
    most_occur = my_counter.most_common(num_words)   
    appearance_on_urls = 0
    for element in most_occur:
        appearance_on_urls = appearance_on_urls + element[1]
    return appearance_on_urls / num_of_words

# calculates what is the portion of the chars of the num_words most common words, from all the urls chars 
def CharsStatistics(num_words):
    urls, _ = UrlReader(0, True)
    urls_tokenize = UrlTokenization(urls)
    list_of_words = []
    num_of_chars = 0
    for url in urls_tokenize:
        for word in url:
            list_of_words.append(word)
            num_of_chars = num_of_chars + len(word)
    my_counter = Counter(list_of_words)
    most_occur = my_counter.most_common(num_words)   
    appearance_on_urls = 0
    for element in most_occur:
        appearance_on_urls = appearance_on_urls + element[1]*len(element[0])
    return appearance_on_urls / num_of_chars 
        
"""
# positinal encoding for the letter's places 
def PositionalEncoding1D(dimension):
    PositionalEncodingVector = torch.zeros(dimension)
    for i in range(0,dimension,2):
        PositionalEncodingVector[i] = math.sin(2/(10_000**((2*i)/dimension))) #to tal: this is how we want to do it????
        PositionalEncodingVector[i+1] = math.cos(2/(10_000**((2*i/dimension))))
    return PositionalEncodingVector

# positinal encoding for the letter's places and the words places
# dim1 is the word place in the sentence, dim2 is the letter place in the word
def PositionalEncoding2D(dim1, dim2):
    PositionalEncodingVector = torch.zeros(dim1, dim2)
    for i in range(dim1):
        for j in range(0,dim2,2):
            PositionalEncodingVector[i,j] = math.sin(dim1/(10_000**((2*j)/dim2))) #to tal: this is how we want to do it????
            PositionalEncodingVector[i,j+1] = math.cos(dim1/(10_000**((2*j/dim2))))
    return PositionalEncodingVector

def LetterToIndex(letter):
    return accepted_chars_after_split.find(letter)

def LetterToTensor(letter, letter_index, word_index, is_1D):
    vector = torch.zeros(n_letters)
    if is_1D:
        vector[LetterToIndex(letter)] = 1 + PositionalEncoding1D[letter_index]
    else:
        vector[LetterToIndex(letter)] = 1 + PositionalEncoding2D[word_index,letter_index]
    return vector

def WordToTensor(word, word_index, is_1D):
    vector = torch.zeros(n_letters)
    for letter_index, letter in enumerate(word):
        one_hot_vec = LetterToTensor(letter, letter_index, word_index, is_1D)
        vector += one_hot_vec
    return vector

# perform the embedding with  
def UrlToTensor(url, max_words_in_url, is_1D):
    tensor = torch.zeros(max_words_in_url,n_letters)
    for word_index, word in enumerate(url):
        tensor[word_index,:] = WordToTensor(word, word_index, is_1D)
    return tensor
"""
