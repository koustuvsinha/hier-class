import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

torch.manual_seed(233)

# Decoder based models
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np

class BLSTM(nn.Module):
    """
    Simple hierarchical decoder. Like fastText, it first encodes the document once,
    and then uses a GRU to predict the categories from a top-down approach.
    """

    def __init__(self, vocab_size=1, embedding_dim=300,
                 category_emb_dim=64, hidden_size=200, label_size=1,
                 pad_token=1, position_size=500, position_dim=50, **kwargs):
        """

        :param vocab_size:
        :param embedding_dim:
        :param category_emb_dim:
        :param total_cats:
        :param label_size: number of total categories
        :param pad_token:
        :param kwargs:
        """
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.position_size = position_size
        self.position_dim = position_dim

        self.pad_token = pad_token
        self.category_emb = category_emb_dim

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        self.position_embedding = nn.Embedding(
            position_size,
            position_dim
        )
        self.category_embedding = nn.Embedding(
            label_size,
            category_emb_dim
        )

        self.word_LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.sent_LSTM = nn.LSTM(
            input_size=hidden_size*2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)

        self.decoder = nn.GRU(category_emb_dim, hidden_size*2, batch_first=True)
        self.decoder2linear = nn.Linear(hidden_size*2, label_size)
        self.logSoftMax = nn.LogSoftmax()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform(-initrange, initrange)
        nn.init.xavier_normal(
            self.decoder2linear.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def encode(self, src, src_lengths):
        """
        Encode the documents
        :param src: documents
        :param src_lengths: length of the documents
        :return:
        """
        def _avg_pooling(self, x): # you need to force all pads to be zero and ignore them when averaging
            result = []
            for i, data in enumerate(x):
                #data = torch.nonzero(data)
               # avg_pooling = torch.mean(torch.nonzero(data).float(), dim=1, keepdim=True)
                avg_pooling = torch.mean(data, dim=1, keepdim=True) #might need to consider zero terms
                result.append(avg_pooling)
            return torch.cat(result, dim=1)

        # unbine based on the sentences
        word_embs_in_sent = [self.embedding(src_sent) for src_sent in torch.unbind(src,1)]
        # a list of sentences, each sentence is 3d tensor with word-hidden states
        words_hiddens_in_sent = [self.word_LSTM(src_emb)[0] for src_emb in word_embs_in_sent]
        #each 3d tensor is avg_pooled with the words aggregate to a sent embedding
        sent_embs_in_doc = _avg_pooling(self, words_hiddens_in_sent)
        #transform from sent embeddings to sentence representations
        sent_hiddens_in_doc = self.sent_LSTM(sent_embs_in_doc)[0] #(b,seq,dim_hidden)
        #break all the sentence hidden state
        sent_hiddens_list = torch.unbind(sent_hiddens_in_doc, 1) #[(b,1,dim_hidden), ...,(b,1,dim_hidden) ]
        # choice one: last sentence hidden states as the rep for the document
        doc_rep = sent_hiddens_list[-1]
        # #choice two: average all sent hidden states
        # doc_rep = torch.mean(sent_hiddens_in_doc,dim=1)
        # #choice three: non-linear transformation of the doc_rep
        # doc_rep = torch.mean(sent_hiddens_in_doc, dim=1)
        # doc_rep = self.tanh(self.fc1(doc_rep))
        return doc_rep

    def forward(self, categories, hidden_state):
        """
        :param src: document to classify
        :param src_lengths: length of the documents
        :param num_cats: number of times to unroll
        :param categories: # keep starting category symbol as 0, such as
                        categories = torch.zeros(batch_size,1)
        :return:
        """
        cat_emb = self.category_embedding(categories)

        output, hidden_state = self.decoder(cat_emb, hidden_state)

        logits = self.decoder2linear(output)
        out = self.logSoftMax(logits.view(-1, self.label_size))
        return out, hidden_state

    def batchNLLLoss(self, src, src_lengths, categories, tf_ratio=1.0):
        """
        Calculate the negative log likelihood loss while predicting the categories
        :param src: documents to be classified
        :param src_lengths: length of the docs
        :param categories: hierarchical categories
        :param tf_ratio: teacher forcing ratio
        :return:
        """
        loss_fn = nn.NLLLoss()
        loss = 0
        accs = []
        hidden_state = self.encode(src, src_lengths).unsqueeze(0)
        cat_len = categories.size(1) - 1
        out = None
        use_tf = True if (random.random() < tf_ratio) else False
        if use_tf:
            for i in range(cat_len):
                inp_cat = categories[:,i]
                inp_cat = inp_cat.unsqueeze(1)
                out, hidden_state = self.forward(inp_cat, hidden_state)
                target_cat = categories[:,i+1]
                loss += loss_fn(out, target_cat)
                _, out_pred = torch.max(out.data, 1)
                acc = (out_pred == target_cat.data).sum() / len(target_cat)
                accs.append(acc)
        else:
            for i in range(cat_len):
                if i == 0:
                    inp_cat = categories[:,i].unsqueeze(1) # starting token
                else:
                    topv, topi = out.data.topk(1)
                    inp_cat = Variable(topi)
                out, hidden_state = self.forward(inp_cat, hidden_state)
                target_cat = categories[:, i+1]
                loss += loss_fn(out, target_cat)
                _, out_pred = torch.max(out.data, 1)
                acc = (out_pred == target_cat.data).sum() / len(target_cat)
                accs.append(acc)

        return loss, np.mean(accs)











