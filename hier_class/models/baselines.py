## Baseline classifiers to be implemented here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class FastText(nn.Module):
    """ Implementation of fasttext in pytorch"""

    def __init__(self, vocab_size=1, embedding_dim=300, label_size=1, pad_token=1, **kwargs):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.pad_token = pad_token
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        self.linear = nn.Linear(embedding_dim, label_size)
        self.logSoftMax = nn.LogSoftmax()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform(-initrange, initrange)
        nn.init.xavier_normal(
            self.linear.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def forward(self, src, src_lengths):
        src_emb = self.embedding(src)
        # average vectors
        src_emb = torch.mean(src_emb, 1)
        logits = self.linear(src_emb)
        out = self.logSoftMax(logits.view(-1, self.label_size))
        return out