from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import numpy as np
import time
import math
import string


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

use_cuda = torch.cuda.is_available()

def indexesFromSentence(sentence):
    '''
     | is our end of sentence character 
     ~ is our start of sentence character
     if these occur naturally in our string, replace it with our unknown character
     _ is our unknown character
    '''
    sentence_as_indices = []
    sentence = sentence.replace("|","_")
    sentence = sentence.replace("~","_")
    for char in sentence:
        if char in char2idx:
            sentence_as_indices.append(char2idx[char])
        else:
            sentence_as_indices.append(char2idx["_"])
            
    sentence_as_indices.append(char2idx["|"])
    
    return sentence_as_indices

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [char2idx[" "] for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size):
    input_seqs = []
    targets = []

    # Choose random pairs
    for i in range(batch_size):
        #pair = random.choice(pairs)
        pair = random.choice(training_pairs)
        input_seqs.append(indexesFromSentence(pair[0]))
        targets.append(class2index[pair[1]])

    # Zip into pairs, sort by length (descending), unzip
    #seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    #input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    #target_lengths = [len(s) for s in target_seqs]
    #target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    #target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    targets = Variable(torch.LongTensor(targets))
    
    if use_cuda:
        input_var = input_var.cuda()
        targets = targets.cuda()
        
    return input_var, input_lengths, targets