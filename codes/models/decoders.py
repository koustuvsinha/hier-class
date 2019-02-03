# Decoder based models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from codes.models.sublayers import DocumentLevelScaledAttention, DocumentLevelSelfAttention
from codes.utils.masked_softmax import MaskedSoftmaxAndLogSoftmax
from codes.utils.model_utils import get_mlp
from codes.utils import constants as Constants
import numpy as np
import time
import pdb

# select device automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.embedding_dim).to(device)
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
                    inp_cat = topi
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


class AttentiveHierarchicalClassifier(nn.Module):
    """
    AttentiveHierarchicalClassifier module
    """

    def __init__(self, vocab_size=1, embedding_dim=300,
                 mlp_hidden_dim=300,
                 cat_emb_dim=64, label_size=1,
                 total_cats=1,pad_token=1,
                 n_layers=1, da=350, n_heads=[8],
                 dropout=0, lstm_dropout=0, hidden_dropout=0,
                 hidden_layers=2,
                 gpu=0,
                 embedding=None,
                 use_embedding=False,
                 fix_embedding=False,
                 multi_class=True,
                 use_rnn=True,
                 prev_emb=False,
                 fix_prev_emb=False,
                 levels=3,
                 pretrained_lm=False,
                 use_parent_emb=False,
                 use_projection=True,
                 label_sizes=[],
                 **kwargs):
        """

        :param vocab_size:
        :param embedding_dim:
        :param mlp_hidden_dim:
        :param cat_emb_dim:
        :param label_size:
        :param total_cats:
        :param pad_token:
        :param n_layers:
        :param da:
        :param n_head:
        :param dropout:
        :param gpu:
        :param embedding:
        :param use_embedding:
        :param fix_embedding:
        :param multi_class:
        :param use_rnn:
        :param kwargs:
        """
        super(AttentiveHierarchicalClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.gpu = gpu
        self.n_layers = n_layers
        self.multi_class = multi_class
        self.use_rnn = use_rnn
        self.n_heads = n_heads
        self.levels = levels
        self.pretrained_lm = pretrained_lm
        self.use_parent_emb = use_parent_emb
        self.label_sizes = label_sizes


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

        self.category_embedding = nn.Embedding(
            total_cats,
            cat_emb_dim
        )

        if fix_prev_emb:
            self.category_embedding.requires_grad = False

        if use_rnn:
            self.encoder = nn.LSTM(embedding_dim, embedding_dim, dropout=lstm_dropout,
                               num_layers=n_layers, bidirectional=True, batch_first=True)

        self.attention = DocumentLevelSelfAttention(embedding_dim, da, n_heads[-1],
                            embedding_dim * 2, cat_emb=cat_emb_dim, use_rnn=self.use_rnn)

        linear_inp = n_heads[-1] * embedding_dim
        if use_rnn:
            linear_inp = n_heads[-1] * embedding_dim * 2
        linear_outp = mlp_hidden_dim
        classifier_inp = mlp_hidden_dim
        if use_parent_emb:
            classifier_inp+= sum(self.label_sizes[:-1])
        if prev_emb:
            #linear_inp = n_heads[-1] * embedding_dim * 2 + mlp_hidden_dim
            classifier_inp += linear_outp
        elif not use_rnn:
            linear_inp = n_heads[-1] * embedding_dim
        else:
            linear_inp = n_heads[-1] * embedding_dim * 2

        self.linear_next = get_mlp(linear_inp, linear_outp, num_layers=hidden_layers, dropout=hidden_dropout)


        if self.multi_class:
            # TODO: need to correct for proper classes in decoder mode
            for i in range(levels):
                setattr(self, 'classifier_l{}'.format(i+1),
                        get_mlp(classifier_inp, label_size, dropout=hidden_dropout))
            self.classifiers = [getattr(self,'classifier_l{}'.format(i + 1)) for i in range(levels)]
        else:
            self.classifier_lall = get_mlp(classifier_inp, label_size)
        self.dropout = nn.Dropout(dropout)
        self.use_projection = use_projection
        if use_projection:
            self.projection = get_mlp(mlp_hidden_dim, cat_emb_dim)
        #self.batchnorm = nn.BatchNorm1d(mlp_hidden_dim)

        self.init_weights()




    def init_weights(self):
        initrange = 0.1
        init.xavier_normal(self.embedding.weight)
        init.xavier_normal(self.category_embedding.weight)
        #self.embedding.weight.data.uniform(-initrange, initrange)
        #self.category_embedding.weight.data.uniform(-initrange, initrange)
        if self.use_rnn:
            for name,param in self.encoder.named_parameters():
                if 'bias' in name:
                    init.constant(param, 0.0)
                elif 'weight' in name:
                    init.xavier_normal(param)

        #if self.multi_class:
        #    for i in range(self.levels):
        #        init.xavier_normal(getattr(self, 'classifier_l{}'.format(i+1)).weight)
        #        init.xavier_normal(getattr(self, 'linear_l{}'.format(i + 1)).weight)
        #else:
        #    init.xavier_normal(self.classifier_lall.weight)
        #    init.xavier_normal(self.linear.weight)

        ## load pretrained weights

        if self.use_rnn and self.pretrained_lm:
            pdb.set_trace()
            lm = torch.load(open('/home/ml/ksinha4/mlp/hier-class/data/lm_pretrained.mod', 'rb'))
            lm_rev = torch.load(open('/home/ml/ksinha4/mlp/hier-class/data/lm_pretrained_reverse.mod', 'rb'))
            for name, param in self.encoder.named_parameters():
                if 'reverse' in name:
                    name = name.replace('_reverse', '')
                param_name = 'rnn.' + name
                # temporary workaround. use backward LM weights here
                if param.size()[-1] > lm[param_name].size()[-1]:
                    param.data = torch.cat((lm[param_name], lm_rev[param_name]),1)
                else:
                    param.data = lm[param_name]
            pdb.set_trace()
            self.encoder.flatten_parameters()
            print('set pretrained language model')





    def init_hidden(self, batch_size, gpu=0):
        hidden = torch.zeros(batch_size, self.mlp_hidden_dim).to(device)
        return hidden


    def encode(self, src, src_lengths):
        """
        Encode the documents
        :param src: documents
        :param src_lengths: length of the documents
        :return:
        """
        src_emb = self.embedding(src)
        #output = torch.mean(src_emb,1)
        src_pack = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        if self.use_rnn:
            src_pack, (h_t, c_t) = self.encoder(src_pack)
        output, output_lens = pad_packed_sequence(src_pack, batch_first=True) # batch x seq x hid

        return output, output_lens

    def forward(self, encoder_outputs, encoder_lens, inp_cat,level=0, prev_emb=None,
                use_prev_emb=False, attn_mask=False, prev_attn=False):
        """

        :param doc_emb:
        :param hidden_state:
        :param prev_emb if not None, then concat category embedding with previous step document embedding
        :return:
        """
        cat_emb = self.category_embedding(inp_cat)
        cat_emb = cat_emb.unsqueeze(1)

        parent_emb = None
        if self.use_parent_emb:
            ## create a parent class embedding layer
            parent_emb = torch.zeros((inp_cat.size(0), sum(self.label_sizes[:-1])))
            for row, inp in enumerate(inp_cat.data.cpu().numpy()):
                parent_emb[row, inp - 1] = 1
            parent_emb = parent_emb.to(device)

        if self.use_projection:
            proj_prev_emb = self.projection(prev_emb).unsqueeze(1)
        else:
            proj_prev_emb = cat_emb

        doc_emb, attn = self.attention(encoder_outputs, encoder_lens,
                                       cat_emb.size(0), proj_prev_emb,prev_attn=prev_attn)

        doc_emb = doc_emb.view(doc_emb.size(0), -1)

        hidden_rep = self.dropout(self.linear_next(doc_emb))
        inter_rep = hidden_rep
        if self.use_parent_emb:
            inter_rep = torch.cat((inter_rep, parent_emb),1)
        if use_prev_emb:
            inter_rep = torch.cat((prev_emb, inter_rep), 1)

        if self.multi_class:
            logits = self.classifiers[level](inter_rep)
        else:
            logits = self.classifier_lall(inter_rep)

        return logits, attn, hidden_rep.view(prev_emb.size())



class PooledHierarchicalClassifier(nn.Module):
    """
    PooledHierarchicalClassifier module
    provides max, mean and concat pooling
    """

    def __init__(self, vocab_size=1, embedding_dim=300,
                 mlp_hidden_dim=300,
                 cat_emb_dim=64, label_size=1,
                 label_sizes=[], label2id={},
                 total_cats=1,pad_token=1,
                 n_layers=1,
                 dropout=0,gpu=0, prev_emb=False,
                 use_attn_mask=False,
                 attention_type='scaled',
                 embedding=None,
                 use_embedding=False,
                 fix_embedding=False,
                 attn_penalty=True,
                 multi_class=True,
                 use_rnn=True,
                 use_cat_emb=False,
                 use_parent_emb=False,
                 pretrained_lm=False,
                 levels=3,
                 **kwargs):
        """


        """
        super(PooledHierarchicalClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.label_sizes = label_sizes
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.label2id = label2id
        self.total_cats = total_cats
        self.gpu = gpu
        self.prev_emb = prev_emb
        self.use_attn_mask = use_attn_mask
        self.attn_penalty = attn_penalty
        self.n_layers = n_layers
        self.multi_class = multi_class
        self.use_rnn = use_rnn
        self.levels = levels
        self.use_cat_emb = use_cat_emb
        self.use_parent_emb = use_parent_emb
        self.pretrained_lm = pretrained_lm
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

        self.category_embedding = nn.Embedding(
            total_cats,
            cat_emb_dim
        )
        # if prev_emb is True, then to use previous embedding make the mult factor = 3
        mult_factor = 1
        if use_rnn:
            mult_factor = 2
            self.encoder = nn.LSTM(embedding_dim, embedding_dim, dropout=dropout,
                               num_layers=n_layers, bidirectional=True, batch_first=True)

        if attention_type == 'concat':
            mult_factor = mult_factor * 2

        self.linear = nn.Linear(embedding_dim * mult_factor, mlp_hidden_dim)
        class_inp = mlp_hidden_dim
        if use_parent_emb:
            class_inp+= sum(self.label_sizes[:-1])
        if prev_emb:
            class_inp += mlp_hidden_dim
        if use_cat_emb:
            class_inp += cat_emb_dim

        if self.multi_class:
            # TODO: need to correct for proper classes in decoder mode
            for i in range(levels):
                setattr(self, 'classifier_l{}'.format(i+1),
                        nn.Linear(class_inp, label_size))
            self.classifiers = [getattr(self,'classifier_l{}'.format(i + 1)) for i in range(levels)]
        else:
            self.classifier_lall = nn.Linear(class_inp, label_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.attention_type = attention_type

        self.init_weights()




    def init_weights(self):
        initrange = 0.1
        init.xavier_normal(self.embedding.weight)
        init.xavier_normal(self.category_embedding.weight)
        #self.embedding.weight.data.uniform(-initrange, initrange)
        #self.category_embedding.weight.data.uniform(-initrange, initrange)
        if self.use_rnn:
            for name,param in self.encoder.named_parameters():
                if 'bias' in name:
                    init.constant(param, 0.0)
                elif 'weight' in name:
                    init.xavier_normal(param)
        init.xavier_normal(self.linear.weight)
        if self.multi_class:
            for i in range(self.levels):
                init.xavier_normal(getattr(self, 'classifier_l{}'.format(i+1)).weight)
        else:
            init.xavier_normal(self.classifier_lall.weight)

        ## load pretrained weights

        if self.use_rnn and self.pretrained_lm:
            pdb.set_trace()
            lm = torch.load(open('/home/ml/ksinha4/mlp/hier-class/data/lm_pretrained.mod', 'rb'))
            lm_rev = torch.load(open('/home/ml/ksinha4/mlp/hier-class/data/lm_pretrained_reverse.mod', 'rb'))
            for name, param in self.encoder.named_parameters():
                if 'reverse' in name:
                    name = name.replace('_reverse', '')
                param_name = 'rnn.' + name
                # temporary workaround. use backward LM weights here
                if param.size()[-1] > lm[param_name].size()[-1]:
                    param.data = torch.cat((lm[param_name], lm_rev[param_name]), 1)
                else:
                    param.data = lm[param_name]
            pdb.set_trace()
            self.encoder.flatten_parameters()
            print('set pretrained language model')




    def init_hidden(self, batch_size, gpu=0):
        hidden = torch.zeros(batch_size, self.mlp_hidden_dim).to(device)
        return hidden


    def encode(self, src, src_lengths):
        """
        Encode the documents
        :param src: documents
        :param src_lengths: length of the documents
        :return:
        """
        src_emb = self.embedding(src)
        #output = torch.mean(src_emb,1)
        src_pack = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        if self.use_rnn:
            src_pack, (h_t, c_t) = self.encoder(src_pack)
        output, output_lens = pad_packed_sequence(src_pack, batch_first=True) # batch x seq x hid

        return output, output_lens

    def forward(self, encoder_outputs, encoder_lens, inp_cat,level=0,
                prev_emb=None,
                use_prev_emb=False,
                use_cat_emb=False,
                attn_mask=False, prev_attn=False):
        """

        :param doc_emb:
        :param hidden_state:
        :param prev_emb if not None, then concat category embedding with previous step document embedding
        :return:
        """
        #pdb.set_trace()
        cat_emb = self.category_embedding(inp_cat)
        cat_emb = cat_emb.unsqueeze(1)

        #pdb.set_trace()
        parent_emb = None
        if self.use_parent_emb:
            ## create a parent class embedding layer
            parent_emb = torch.zeros((inp_cat.size(0), sum(self.label_sizes[:-1])))
            for row, inp in enumerate(inp_cat.data.cpu().numpy()):
                parent_emb[row, inp-1] = 1
            parent_emb = parent_emb.to(device)

        if self.attention_type == 'maxpool':
            # Maxpool
            doc_emb = torch.max(encoder_outputs, 1)[0]
            # or mean pool
            # doc_emb = torch.mean(encoder_outputs, 1).squeeze()
            attn = None
        elif self.attention_type == 'meanpool':
            doc_emb = torch.mean(encoder_outputs, 1).squeeze()
        elif self.attention_type == 'concat':
            #pdb.set_trace()
            maxp = torch.max(encoder_outputs, 1)[0]
            meanp = torch.mean(encoder_outputs, 1)
            doc_emb = torch.cat((maxp, meanp),1)
        else:
            raise NotImplementedError("attention type not implemented")

        #pdb.set_trace()
        #print(doc_emb.size())
        doc_emb = doc_emb.view(doc_emb.size(0), -1)
        hidden_rep = self.linear(doc_emb)
        inter_rep = hidden_rep
        if self.use_parent_emb:
            inter_rep = torch.cat((inter_rep, parent_emb),1)
        if use_prev_emb:
            inter_rep = torch.cat((prev_emb, inter_rep), 1)
        if self.use_cat_emb:
            inter_rep = torch.cat((cat_emb.squeeze(1), inter_rep), 1)

        if self.multi_class:
            logits = self.classifiers[level](self.relu(inter_rep))
        else:
            logits = self.classifier_lall(self.relu(inter_rep))

        return logits, None, hidden_rep.view(prev_emb.size())

class Trainer():
    """
    Trainer instance which takes in any above model and runs training
    and inference
    """
    def __init__(self,
                 model,
                 loss_focus=[],
                 loss_weights=None,
                 label_sizes=[],
                 n_heads=[],
                 label2id={},
                 renormalize='level',
                 max_categories=3,
                 batch_masking=False,
                 confidence_threshold=0.8,
                 attn_penalty_coeff=0,
                 use_attn_mask=False,
                 prev_emb=False,
                 target_level=-1,
                 total_cats=0,
                 temperature=1,
                 taxonomy=None,
                 max_words=0,
                 detach_encoder=False,
                 teacher_forcing=True,
                 **kwargs
                 ):
        self.model = model
        self.loss_focus = loss_focus
        self.loss_weights = loss_weights
        self.renormalize = renormalize
        self.max_categories = max_categories
        self.batch_masking = batch_masking
        self.confidence_threshold = confidence_threshold
        self.attn_penalty_coeff = attn_penalty_coeff
        self.use_attn_mask = use_attn_mask
        self.target_level = target_level
        self.use_prev_emb = prev_emb
        self.total_cats = total_cats
        self.temperature = temperature
        self.taxonomy = taxonomy
        self.max_words = max_words
        self.label_sizes = label_sizes
        self.label2id = label2id
        self.n_heads = n_heads
        self.detach_encoder = detach_encoder
        self.teacher_forcing = teacher_forcing
        if type(loss_weights) == torch.FloatTensor:
            self.loss_fn = nn.NLLLoss(weight=loss_weights)
        else:
            self.loss_fn = nn.NLLLoss()

    def batchNLLLoss(self, src, src_lengths, categories, mode='train', overall=True, tf_ratio=1):
        """
        Calculate the negative log likelihood loss while predicting the categories
        :param src: documents to be classified
        :param src_lengths: length of the docs
        :param categories: hierarchical categories
        :return:
        """

        loss = 0
        log_loss = 0
        accs = []
        encoder_outputs, encoder_lens = self.model.encode(src, src_lengths)
        hidden_rep = self.model.init_hidden(src.size(0))
        cat_len = categories.size(1) - 1
        # assert cat_len == max_categories
        out = None
        out_p = None
        level_cs = []
        attns = []
        probs = []
        correct_labels = []
        predictions = []
        correct_confs = []
        incorrect_confs = []
        levels = len(self.label_sizes)

        #pdb.set_trace()
        prev_attns = [torch.ones(
            self.n_heads[-1], encoder_lens[b]).to(device)
                      for b in range(encoder_outputs.size(0))]

        # if overall is set to true (by default)
        # if mode is train, then either train with teacher forcing or not
        # depending on the experiment
        # if mode is inference, then teacher forcing should be:
        #   true if overall is false
        #   false if overall is true
        # this gives us a way to run overall = True and overall = False on each
        #   validation inference
        teacher_forcing = self.teacher_forcing
        if mode != 'train':
            if overall:
                teacher_forcing = False
            else:
                teacher_forcing = True
        else:
            teacher_forcing = True if (random.random() < tf_ratio) else False


        # training or inference
        for i in range(levels):
            # detach encoder
            if self.detach_encoder:
                if i > 0:
                    encoder_outputs = encoder_outputs.detach()
            if teacher_forcing or i == 0: # or
                inp_cat = categories[:, i]
            else:
                topv, topi = out.data.topk(1)
                inp_cat = topi.squeeze(1)
            if torch.max(inp_cat).data.cpu().numpy() > self.total_cats:
                print(inp_cat)
                raise RuntimeError("category ID outside of embedding")
            # hidden_state = torch.cat((hidden_state, context_state), 2)
            #inp_cat = inp_cat.unsqueeze(1)
            if self.use_attn_mask:
                attn_mask = self.get_attn_padding_mask(inp_cat, src)
            else:
                attn_mask = None
            out, attn, hidden_rep = self.model(encoder_outputs, encoder_lens,
                                            inp_cat, i, prev_emb=hidden_rep,
                                            use_prev_emb=self.use_prev_emb,
                                            attn_mask=attn_mask,
                                            prev_attn=prev_attns)
            prev_attns = attn
            log_sum = torch.mean(torch.sum(out, dim=1))
            if self.renormalize:
                if self.renormalize == 'level':
                    out, log_sum = self.mask_level(out,i)
                elif self.renormalize == 'category':
                    out, log_sum = self.mask_category(out,inp_cat)

            temp = 1
            if i > 0:
                temp = self.temperature
            out = self.temp_logsoftmax(out, temp)
            prob = torch.exp(out)
            target_cat = categories[:, i+1]
            if self.attn_penalty_coeff > 0:
                attn_penalty = self.calculate_attention_penalty(attn, batch_size=inp_cat.size(0))
            else:
                attn_penalty = 0
            # calculate loss
            #pdb.set_trace()
            log_loss += log_sum
            loss += self.loss_fn(out, target_cat) * self.loss_focus[i] + \
                    self.attn_penalty_coeff * attn_penalty
            #out = self.mask_renormalize(inp_cat, out)
            pred_logits, out_pred = torch.max(out.data, 1)

            correct_idx = (out_pred == target_cat.data)
            incorrect_idx = 1 - correct_idx
            acc = correct_idx.float().mean().item()
            # check if atleast one of them is correct
            if correct_idx.any():
                correct_pred_conf = torch.exp(pred_logits[correct_idx]).float().mean().item()
            else:
                correct_pred_conf = 0.0

            if incorrect_idx.any():
                incorrect_conf = torch.exp(pred_logits[incorrect_idx]).float().mean().item()
            else:
                incorrect_conf = 0.0

            accs.append(acc)
            attns.append(self.convert_cpu(attn))
            probs.append(self.convert_cpu(prob))
            predictions.append(out_pred.cpu().numpy())
            correct_labels.append(target_cat.data.cpu().numpy())
            correct_confs.append(correct_pred_conf)
            incorrect_confs.append(incorrect_conf)

        log_loss = log_loss / src.size(0)

        return (loss, log_loss), accs, attns, predictions, correct_labels, correct_confs, incorrect_confs, probs

    def apply_softmax(self, xs, mask, dtype=torch.DoubleTensor):
        return MaskedSoftmaxAndLogSoftmax(dtype)(xs, mask)

    def mask_level(self, logits, level=0):
        """
        Given level, mask out all the other level classes
        :param parent_class_batch: parent class ID in batch, batch x 1
        :param logits: batch x classes
        :return:
        """
        mask = [0]*(sum(self.label_sizes) + 1) # 912
        label_indices = {}
        ct = 1
        for lv,lbs in enumerate(self.label_sizes):
            label_indices[lv] = []
            for lb in range(lbs):
                label_indices[lv].append(ct)
                ct +=1
        for indc in label_indices[level]:
            mask[indc] = 1
        row_mask = [mask for i in range(logits.size(0))]
        #pdb.set_trace()
        #
        #print(mask)
        #row_mask = Variable(torch.ByteTensor(row_mask).cuda()).double()
        mask = torch.ByteTensor(mask).to(device)
        mask = mask ^ 1
        #probs, logits = self.apply_softmax(logits.double(), row_mask)
        logits.data.masked_fill_(mask, 0)
        log_sum = torch.mean(torch.sum(logits, dim=1))
        logits.data.masked_fill_(mask, -float('inf'))
        return logits, log_sum

    def mask_category(self, logits, parent_class_batch, loss_weights=None):
        """
        Given a parent class, logits and taxonomy, mask the classes which are not in child
        :param logits:
        :param level:
        :return:
        """
        mask = torch.ones(logits.size())
        parent_class_batch = parent_class_batch.data.cpu().numpy()
        for batch_id, parent_class in enumerate(parent_class_batch):
            if parent_class in self.taxonomy:
                child_classes = self.taxonomy[parent_class]
                for cc in child_classes:
                    mask[batch_id][cc] = 0
        #pdb.set_trace()
        #mask = mask.byte() ^ 1
        #mask = Variable(mask.float()).detach()
        #mask = mask.cuda()
        #probs = torch.exp(logits)
        #masked_probs = probs * mask
        # renormalize
        #_sums = masked_probs.sum(-1, keepdim=True)
        #probs = masked_probs.div(_sums)
        #logits = torch.log(probs)
        mask = mask.byte().to(device)
        logits.data.masked_fill_(mask, 0)
        # pdb.set_trace()
        log_sum = torch.mean(torch.sum(logits, dim=1))
        logits.data.masked_fill_(mask, -float('inf'))
        return logits, log_sum

    def calculate_loss(self):
        """
        For the correct class, do NLLLoss like usual
        For the classes which cannot be present, run NLLLoss for each
        class and subtract it from the loss.
        Also, average those log likelihoods by the number of impossible classes?
        :return:

        """
        pass

    def calculate_attention_penalty(self, attns, batch_size):
        """
        From Self attentive paper, use similar Frobenius norm penalty to separate attentions
        :param attns : (n_headxbatch, 1, seq?)
        :return:
        """
        penalty = 0
        if type(attns) == list:
            n_heads = attns[0].size(0)
        else:
            attns = attns.view(batch_size, -1, attns.size(2))
            n_heads = attns.size(1)
        I = torch.eye(n_heads).to(device)
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

    def temp_logsoftmax(self, y, temperature):
        return F.log_softmax(y / temperature, dim=-1)

    def label2category(self, label, level):
        return self.label2id['l{}_{}'.format(label, level)]

    def convert_cpu(self, attn):
        if type(attn) == list:
            attn = [a.data.cpu().numpy() for a in attn]
        else:
            attn = attn.data.cpu().numpy()
        return attn


    def get_attn_padding_mask(self, seq_q, seq_k):
        ''' Indicate the padding-related part to mask '''
        assert seq_q.dim() == 1 and seq_k.dim() == 2
        mb_size = seq_q.size()
        len_q = 1
        mb_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
        pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
        return pad_attn_mask






