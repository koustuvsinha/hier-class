import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from hier_class.models.modules import ScaledDotProductAttention, LayerNormalization
from hier_class.utils import constants as Constants
import pdb

class DocumentLevelAttention(nn.Module):
    """
    Document Level Attention Layer
    Have provision for multiple heads
    """
    def __init__(self, d_model, d_k, d_v, n_head=1, dropout=0, bidirectional=True):
        super(DocumentLevelAttention, self).__init__()
        if bidirectional:
            d_model = d_model * 2
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, encoder_lens=None, attn_mask=None, atm=True):
        '''

        :param q: query - category embedding
        :param k: key - encoder outputs
        :param v: value - encoder outputs
        :param encoder_lens: length of the encoded values
        :param attn_mask: inverse mask. for valid elements its 0, for padding its 1.
        :return:
        '''
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if atm:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        else:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class DocumentLevelSelfAttention(nn.Module):
    """
    Heavily inspired from https://github.com/yufengm/SelfAttentive/blob/master/model.py
    :param nhid: encoder hidden dimension
    :param da: hidden dim of S1
    :r :number of hops
    :mlp_nhid: output of mlp dimension
    """
    def __init__(self, nhid, da, r, mlp_nhid, cuda=True):
        super(DocumentLevelSelfAttention, self).__init__()
        self.S1 = nn.Linear(nhid * 3, da, bias=False)
        self.S2 = nn.Linear(da, r, bias=False)
        self.MLP = nn.Linear(r * nhid * 3, mlp_nhid)

        self.r = r
        self.nhid = nhid
        self.cuda = cuda

    def init_weights(self):
        initrange = 0.1
        self.S1.weight.data.uniform_(-initrange, initrange)
        self.S2.weight.data.uniform_(-initrange, initrange)

        self.MLP.weight.data.uniform_(-initrange, initrange)
        self.MLP.bias.data.fill_(0)

    def forward(self, encoder_outputs, encoder_lengths, batch_size, cat_emb, temp=1):
        """
        n = max length of sequence
        D = dimension of model
        B = batch
        :param encoder_outputs: B x n x 2D = H
        :param encoder_lengths: B x 1
        :param batch_size: B
        :param cat_emb: B x 1 x D, V
        :param temp: temperature for softmax
        :return:

        Original self attention: A = softmax(W_{s2} tanh(W_{s_1} H^T))
        where,
            W_{s_1} = d_a x 2 D
            W_{s_2} = r x d_a # no. of lookups
        Bahdanau style self attention: concat the category embedding on top of each row of H
            Same equation, \bar{H} = H (+) V = B x n x 3D
        Changes required,
            W_{s_1} = d_a x 3 D
        """
        BM = Variable(torch.zeros(batch_size, self.r * self.nhid * 3))
        if self.cuda:
            BM = BM.cuda()
        weights = []
        HV = encoder_outputs
        HV = torch.cat([HV, cat_emb.expand(cat_emb.size(0), HV.size(1), cat_emb.size(2))], 2)

        for i in range(batch_size):
            H = HV[i, :encoder_lengths[i],:] # n x 3D
            s1 = self.S1(H) # n x da
            s2 = self.S2(F.tanh(s1)) # n x r
            A = F.softmax(s2.t() / temp, dim=1) # r x n
            M = torch.mm(A,H) # (r x n) * (n x 3D) = r x 3D
            BM[i,:] = M.view(-1)
            weights.append(A)

        out = self.MLP(BM)
        return out, weights


def pad(variable, length):
    return torch.cat([variable, variable.new(length - variable.size(0), *variable.size()[1:]).zero_()])


