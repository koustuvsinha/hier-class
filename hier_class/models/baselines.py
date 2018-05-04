## Baseline classifiers to be implemented here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

    def batchNLLLoss(self, src, src_lengths, labels, target_level=0, label_weights=None, **kwargs):
        loss_fn = nn.NLLLoss(label_weights)
        target_cat = labels[:, target_level]
        out = self.forward(src, src_lengths)
        loss = loss_fn(out, target_cat)
        _, out_pred = torch.max(out.data, 1)
        acc = (out_pred == target_cat.data).float().mean()
        return loss, [acc], None, [out_pred.cpu().numpy()], [target_cat.data.cpu().numpy()]

class BiLSTM_MLP(nn.Module):
    def __init__(self, embedding_dim=300, vocab_size=1, pad_token=0, hidden_dim=3000, label_size=1,
                 temperature=1, **kwargs):
        super(BiLSTM_MLP, self).__init__()

        self.temperature = temperature
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        self.encoder = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(embedding_dim*2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, label_size)
        self.relu = nn.ReLU()


    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform(-initrange, initrange)
        nn.init.xavier_normal(
            self.decoder2linear.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def temp_logsoftmax(self, y, temperature=1):
        return F.log_softmax(y / temperature, dim=-1)

    def forward(self, src, src_lengths):
        src_emb = self.embedding(src)
        # output = torch.mean(src_emb,1)
        src_pack = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        packed_output, (h_t, c_t) = self.encoder(src_pack)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # batch x seq x hid
        ## max pooling
        doc_rep = torch.max(output, 1)[0] # batch x hid
        x = self.linear1(doc_rep)
        x = self.relu(x)
        x = self.linear2(x)
        logits = self.temp_logsoftmax(x, self.temperature)
        return logits

    def batchNLLLoss(self, src, src_lengths, labels, target_level=0, **kwargs):
        loss_fn = nn.NLLLoss()
        target_cat = labels[:, target_level]
        out = self.forward(src, src_lengths)
        loss = loss_fn(out, target_cat)
        _, out_pred = torch.max(out.data, 1)
        acc = (out_pred == target_cat.data).float().mean()
        return loss, [acc], None, [out_pred.cpu().numpy()], [target_cat.data.cpu().numpy()]






