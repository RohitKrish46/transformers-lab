import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
import pandas as pd

## Setting up dataset
lines = open('./input.txt', 'r').read()

vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}

# Simple Charecter Level Tokenizer

def encode(s):
    return [stoi[ch] for ch in s]

def decode(s):
    return ''.join([itos[i] for i in s])

# Parameter store
MASTER_CONFIG = {
    "vocab_size": len(vocab),
    'batch_size': 8,
    'context_window': 16,
    'd_model': 128
}

# Encode the entire dataset
dataset = torch.tensor(encode(lines), dtype=torch.int8)

# Function to generate our training data and labels for batches
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]
    
    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test
    
    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y