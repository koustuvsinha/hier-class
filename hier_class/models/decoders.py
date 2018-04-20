# Decoder based models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from hier_class.models.sublayers import DocumentLevelAttention, DocumentLevelSelfAttention
from hier_class.utils import constants as Constants
import numpy as np
import time
import pdb


class SimpleDecoder(nn.Module):
    """
    Simple hierarchical decoder. Like fastText, it first encodes the document once,
    and then uses a GRU to predict the categories from a top-down approach.
    """

    def __init__(self, vocab_size=1, embedding_dim=300, cat_emb_dim=64, label_size=1, pad_token=1,
                 temperature=0.8,**kwargs):
        """

        :param vocab_size:
        :param embedding_dim:
        :param category_emb_dim:
        :param total_cats:
        :param label_size: number of total categories
        :param pad_token:
        :param kwargs:
        """
        super(SimpleDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.pad_token = pad_token
        self.category_emb = cat_emb_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        self.category_embedding = nn.Embedding(
            label_size,
            self.category_emb
        )
        self.decoder = nn.GRU(self.category_emb, embedding_dim, batch_first=True, dropout=0.5)
        #self.hidden2next = nn.Linear(embedding_dim*2, embedding_dim)
        self.decoder2linear = nn.Linear(embedding_dim, label_size)
        #self.logSoftMax = nn.LogSoftmax()
        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform(-initrange, initrange)
        nn.init.xavier_normal(
            self.decoder2linear.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def init_hidden(self, batch_size, gpu=0):
        hidden = Variable(torch.zeros(1, batch_size, self.embedding_dim)).cuda(gpu)
        return hidden

    def temp_logsoftmax(self, y, temperature):
        return F.log_softmax(y / temperature, dim=-1)


    def encode(self, src, src_lengths):
        """
        Encode the documents
        :param src: documents
        :param src_lengths: length of the documents
        :return:
        """
        src_emb = self.embedding(src)
        src_emb = torch.mean(src_emb,1)
        return src_emb

    def forward(self, categories, context_emb, hidden_state):
        """
        :param src: document to classify
        :param src_lengths: length of the documents
        :param num_cats: number of times to unroll
        :param categories: # keep starting category symbol as 0, such as
                        categories = torch.zeros(batch_size,1)
        :param context: embedding of context
        :return:
        """
        cat_emb = self.category_embedding(categories)

        #combined = torch.cat((cat_emb, context_emb.transpose(0,1)), -1)
        output, hidden_state = self.decoder(cat_emb, hidden_state)

        # combined = torch.cat((output, context_emb.transpose(0,1)), -1)
        # output = self.hidden2next(output)
        logits = self.decoder2linear(output)
        out = self.temp_logsoftmax(logits.view(-1, self.label_size), self.temperature)
        return out, hidden_state

    def batchNLLLoss(self, src, src_lengths, categories, tf_ratio=1.0, loss_focus=[],
                     loss_weights=None):
        """
        Calculate the negative log likelihood loss while predicting the categories
        :param src: documents to be classified
        :param src_lengths: length of the docs
        :param categories: hierarchical categories
        :param tf_ratio: teacher forcing ratio
        :return:
        """
        loss_fn = nn.NLLLoss(weight=loss_weights)
        loss = 0
        accs = []
        context_state = self.encode(src, src_lengths).unsqueeze(0)
        hidden_state = context_state #self.init_hidden(context_state.size(1))
        cat_len = categories.size(1) - 1
        #print("Categories : {}".format(cat_len))
        assert cat_len == 3
        out = None
        use_tf = True if (random.random() < tf_ratio) else False
        if use_tf:
            for i in range(cat_len):
                inp_cat = categories[:,i]
                inp_cat = inp_cat.unsqueeze(1)
                #hidden_state = torch.cat((hidden_state, context_state), 2)
                out, hidden_state = self.forward(inp_cat, context_state, hidden_state)
                target_cat = categories[:,i+1]
                loss += loss_fn(out, target_cat) * loss_focus[i]
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
                #hidden_state = torch.cat((hidden_state, context_state), 2)
                out, hidden_state = self.forward(inp_cat, context_state, hidden_state)
                target_cat = categories[:, i+1]
                loss += loss_fn(out, target_cat) * loss_focus[i]
                _, out_pred = torch.max(out.data, 1)
                acc = (out_pred == target_cat.data).sum() / len(target_cat)
                accs.append(acc)

        return loss, accs

    def hierarchical_loss(self):
        """
        Calculate hierarchical loss. The idea is, if the current token is among improbable tokens,
        then return a high negative value. But it seems like this is an ideal case for REINFORCE
        :return:
        """
        pass


class SimpleMLPDecoder(nn.Module):
    """
    Simple hierarchical MLP decoder. Doesn't use a GRU, instead uses an MLP to classify per step.
    """

    def __init__(self, vocab_size=1, embedding_dim=300,
                 mlp_hidden_dim=300,
                 cat_emb_dim=64, label_size=1,
                 label_sizes=[], label2id={},
                 total_cats=1,pad_token=1, max_categories=1,
                 taxonomy=None,temperature=0.8, max_words=600,
                 d_k=64, d_v=64,da=350, n_heads=[2,2,8], dropout=0,gpu=0, prev_emb=False, top_level_cat=0,
                 use_attn_mask=False,
                 attention_type='scaled',
                 embedding=None,
                 use_embedding=False,
                 fix_embedding=False,
                 **kwargs):
        """

        :param vocab_size:
        :param embedding_dim:
        :param cat_emb_dim:
        :param label_size:
        :param label_sizes:
        :param label2id:
        :param total_cats: total number of categories
        :param pad_token: 0
        :param taxonomy: tree structure of external KB
        :param temperature: log softmax temperature
        :param max_words: max words per document
        :param d_k: dimension of Key for attention
        :param d_v: dimension of Value for attention
        :param n_heads: number of heads for attention for each level. default 2,2,8
        :param dropout: default 0.1
        :param gpu: gpu id, default 0 (if using CUDA_VISIBLE_DEVICES then no need to use this)
        :param prev_emb: if True use previous embedding
        :param attention_type: scaled or self
        :param kwargs:
        """
        super(SimpleMLPDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.label_sizes = label_sizes
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.temperature = temperature
        self.label2id = label2id
        self.total_cats = total_cats
        self.gpu = gpu
        self.prev_emb = prev_emb
        self.use_attn_mask = use_attn_mask
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        if use_embedding:
            print("Setting pretrained embedding")
            self.embedding.weight.data = embedding
            if fix_embedding:
                self.embedding.requires_grad = False
        # for attention to work, embedding_dim and category embedding dim should be the same
        if (embedding_dim * 2) != cat_emb_dim:
            print(embedding_dim)
            print(cat_emb_dim)
            raise RuntimeError("for attention to work, embedding_dim and category embedding dim should be the same or double for bidirectional")
        self.category_embedding = nn.Embedding(
            total_cats,
            cat_emb_dim
        )
        # if prev_emb is True, then to use previous embedding make the mult factor = 3
        mult_factor = 2
        if prev_emb:
            mult_factor = 3
        self.encoder = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(embedding_dim * mult_factor, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, label_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #self.linear1 = nn.Linear(embedding_dim + cat_emb_dim, self.label_sizes[0])
        #self.linear2 = nn.Linear(embedding_dim + cat_emb_dim, self.label_sizes[1])
        #self.linear3 = nn.Linear(embedding_dim + cat_emb_dim, self.label_sizes[2])
        #self.linears = [self.linear1, self.linear2, self.linear3]
        self.taxonomy = taxonomy
        self.max_words = max_words
        self.attention_type = attention_type
        if attention_type == 'scaled':
            for i in range(max_categories):
                setattr(self, 'attention_{}'.format(i+1),
                        DocumentLevelAttention(embedding_dim,d_k,d_v, n_head=n_heads[i],
                                                      dropout=dropout))
            self.attentions = [getattr(self, 'attention_{}'.format(i+1)) for i in range(max_categories)]
        elif attention_type == 'self':
            for i in range(max_categories):
                setattr(self, 'attention_{}'.format(i+1),
                        DocumentLevelSelfAttention(embedding_dim,da,n_heads[i],embedding_dim * 2))
            self.attentions = [getattr(self,'attention_{}'.format(i + 1)) for i in range(max_categories)]
        else:
            return



    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform(-initrange, initrange)
        self.category_embedding.weight.data.uniform(-initrange, initrange)
        nn.init.xavier_normal(
            self.decoder2linear.weight,
            gain=nn.init.calculate_gain('tanh')
        )

    def init_hidden(self, batch_size, gpu=0):
        hidden = Variable(torch.zeros(batch_size, self.embedding_dim)).cuda(gpu)
        return hidden

    def temp_logsoftmax(self, y, temperature):
        return F.log_softmax(y / temperature, dim=-1)


    def encode(self, src, src_lengths):
        """
        Encode the documents
        :param src: documents
        :param src_lengths: length of the documents
        :return:
        """
        # TODO: use Attention - fix max doc length
        src_emb = self.embedding(src)
        #output = torch.mean(src_emb,1)
        src_pack = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        packed_output, (h_t, c_t) = self.encoder(src_pack)
        output, output_lens = pad_packed_sequence(packed_output, batch_first=True) # batch x seq x hid

        return output, output_lens

    def forward(self, encoder_outputs, encoder_lens, inp_cat,level=0, prev_emb=None, use_prev_emb=False, attn_mask=False):
        """

        :param doc_emb:
        :param hidden_state:
        :param prev_emb if not None, then concat category embedding with previous step document embedding
        :return:
        """
        cat_emb = self.category_embedding(inp_cat)
        cat_emb = cat_emb.unsqueeze(1)
        if self.attention_type == 'scaled':
            doc_emb, attn = self.attentions[level](cat_emb, encoder_outputs, encoder_outputs,
                                                   attn_mask=None, atm=None)
            doc_emb = doc_emb.squeeze(1)
        elif self.attention_type == 'self':
            doc_emb, attn = self.attentions[level](encoder_outputs, encoder_lens, cat_emb.size(0))
        elif self.attention_type == 'no_attention':
            # Maxpool
            doc_emb = torch.max(encoder_outputs, 1)[0]
            attn = None
        else:
            raise NotImplementedError("attention type not implemented")
        if use_prev_emb:
            doc_emb = torch.cat((prev_emb, doc_emb), 1)
        x = self.linear(doc_emb)
        hidden_rep = self.dropout(x)
        x = self.relu(hidden_rep)
        x = self.linear2(x)
        x = self.dropout(x)
        #x = self.linears[level](x)
        logits = self.temp_logsoftmax(x, self.temperature)

        return logits, attn, hidden_rep

    def batchNLLLoss(self, src, src_lengths, categories, tf_ratio=1.0, loss_focus=[],
                     loss_weights=None, renormalize=False, max_categories=3, batch_masking=False,
                     confidence_threshold=0.8, attn_penalty_coeff=0,
                     target_level=-1):
        """
        Calculate the negative log likelihood loss while predicting the categories
        :param src: documents to be classified
        :param src_lengths: length of the docs
        :param categories: hierarchical categories
        :param tf_ratio: teacher forcing ratio
        :return:
        """
        if type(loss_weights) == torch.FloatTensor:
            loss_fn = nn.NLLLoss(weight=loss_weights)
        else:
            loss_fn = nn.NLLLoss()
        #loss_fns = [nn.NLLLoss(), nn.NLLLoss(), nn.NLLLoss()]
        loss = 0
        accs = []
        encoder_outputs, encoder_lens = self.encode(src, src_lengths)
        hidden_rep = self.init_hidden(src.size(0))
        cat_len = categories.size(1) - 1
        assert cat_len == max_categories
        out = None
        out_p = None
        #pdb.set_trace()
        level_cs = []
        attns = []
        correct_labels = []
        predictions = []
        correct_confs = []
        incorrect_confs = []

        use_tf = True if (random.random() < tf_ratio) else False
        if use_tf:
            for i in range(cat_len):
                inp_cat = categories[:, i]
                if torch.max(inp_cat).data.cpu().numpy() > self.total_cats:
                    print(inp_cat)
                    raise RuntimeError("category ID outside of embedding")
                # hidden_state = torch.cat((hidden_state, context_state), 2)
                #inp_cat = inp_cat.unsqueeze(1)
                attn_mask = None #get_attn_padding_mask(inp_cat, src)
                out, attn, hidden_rep = self.forward(encoder_outputs, encoder_lens, inp_cat, i, prev_emb=hidden_rep,
                                                     use_prev_emb=self.prev_emb,attn_mask=attn_mask)
                if renormalize:
                    out = self.mask_renormalize(inp_cat, out)
                target_cat = categories[:, i+1]
                attn_penalty = 0 #self.calculate_attention_penalty(attn, batch_size=inp_cat.size(0))
                loss += loss_fn(out, target_cat) * loss_focus[i] + attn_penalty_coeff * attn_penalty
                #out = self.mask_renormalize(inp_cat, out)
                pred_logits, out_pred = torch.max(out.data, 1)
                correct_idx = (out_pred == target_cat.data)
                incorrect_idx = 1 - correct_idx
                acc = correct_idx.float().mean()
                # check if atleast one of them is correct
                if pred_logits[correct_idx].dim() > 0:
                    correct_pred_conf = torch.exp(pred_logits[correct_idx]).float().mean()
                else:
                    correct_pred_conf = 0.0

                if pred_logits[incorrect_idx].dim() > 0:
                    incorrect_conf = torch.exp(pred_logits[incorrect_idx]).float().mean()
                else:
                    incorrect_conf = 0.0

                accs.append(acc)
                attns.append(attn)
                predictions.append(out_pred.cpu().numpy())
                correct_labels.append(target_cat.data.cpu().numpy())
                correct_confs.append(correct_pred_conf)
                incorrect_confs.append(incorrect_conf)
        else:
            if batch_masking:
                batch_mask = torch.ones(encoder_outputs.size(0)).long()
                batch_mask = batch_mask.cuda(self.gpu)
            else:
                batch_mask = None
            last_p = 1
            for i in range(cat_len):
                #pdb.set_trace()
                if i == 0:
                    inp_cat = categories[:, i]  # starting token
                else:
                    topv, topi = out.data.topk(1)
                    inp_cat = Variable(topi).squeeze(1)
                if torch.max(inp_cat).data.cpu().numpy() > self.total_cats:
                    print(inp_cat)
                    print(topi)
                    print(out.size())
                    raise RuntimeError("category ID outside of embedding")
                #inp_cat = categories[:, i]
                #inp_cat = inp_cat.unsqueeze(1)
                attn_mask = None #get_attn_padding_mask(inp_cat, src)
                out, attn, hidden_rep = self.forward(encoder_outputs, encoder_lens, inp_cat, i, prev_emb=hidden_rep,
                                                     use_prev_emb=self.prev_emb,attn_mask=attn_mask)
                if renormalize:
                    out = self.mask_renormalize(inp_cat, out)
                target_cat = categories[:, i + 1]
                if batch_masking:
                    target_cat = target_cat * Variable(batch_mask)
                    out = (out.transpose(0,1) * Variable(batch_mask.float())).transpose(0,1)
                attn_penalty = 0 #self.calculate_attention_penalty(attn, batch_size=inp_cat.size(0))
                loss += loss_fn(out, target_cat) * loss_focus[i] + attn_penalty_coeff * attn_penalty
                #out = self.mask_renormalize(inp_cat, out)
                pred_logits, out_pred = torch.max(out.data, 1)
                correct_idx = (out_pred == target_cat.data)
                incorrect_idx = 1 - correct_idx
                acc = correct_idx.float().mean()
                if pred_logits[correct_idx].dim() > 0:
                    correct_pred_conf = torch.exp(pred_logits[correct_idx]).float().mean()
                else:
                    correct_pred_conf = 0.0

                if pred_logits[incorrect_idx].dim() > 0:
                    incorrect_conf = torch.exp(pred_logits[incorrect_idx]).float().mean()
                else:
                    incorrect_conf = 0.0
                #print(out_pred)
                #print(target_cat.data)
                if batch_masking:
                    all_confs = torch.exp(pred_logits).float()
                    level_correct = (all_confs > confidence_threshold).long()
                    batch_mask = batch_mask * level_correct

                #acc = last_p * acc
                #last_p = acc
                #level_cs.append(torch.sum(level_correct))

                accs.append(acc)
                attns.append(attn)
                predictions.append(out_pred.cpu().numpy())
                correct_labels.append(target_cat.data.cpu().numpy())
                correct_confs.append(correct_pred_conf)
                incorrect_confs.append(incorrect_conf)

        return loss, accs, attns, predictions, correct_labels, correct_confs, incorrect_confs

    def mask_renormalize(self, parent_class_batch, logits):
        """
        Given a parent class, logits and taxonomy, mask the classes which are not in child
        :param parent_class_batch: parent class ID in batch, batch x 1
        :param logits: batch x classes
        :return:
        """
        mask = torch.ones(logits.size())
        parent_class_batch = parent_class_batch.data.cpu().numpy()
        for batch_id, parent_class in enumerate(parent_class_batch):
            if parent_class in self.taxonomy:
                child_classes = self.taxonomy[parent_class]
                for cc in child_classes:
                    mask[batch_id][cc] = 0
        mask = mask.byte().cuda(self.gpu)
        logits.masked_fill_(mask, -float('inf'))
        return logits

    def calculate_attention_penalty(self, attns, batch_size):
        """
        From Self attentive paper, use similar Frobenius norm penalty to separate attentions
        :param attns : (n_headxbatch, 1, seq?)
        :return:
        """
        penalty = 0
        if type(attns) == Variable:
            attns = attns.view(batch_size, -1, attns.size(2))
            n_heads = attns.size(1)
        elif type(attns) == list:
            n_heads = attns[0].size(0)
        else:
            return 0
        I = Variable(torch.eye(n_heads)).cuda()
        for i in range(batch_size):
            if type(attns) == list:
                A = attns[i]
            else:
                A = attns[i,:,:]
            AAT = torch.mm(A, A.t())
            P = torch.norm(AAT - I, 2)
            penalty += P * P
        penalty = penalty / batch_size
        return penalty



    def label2category(self, label, level):
        return self.label2id['l{}_{}'.format(label, level)]


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    #print(seq_q.size())
    #print(seq_k.size())
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask






