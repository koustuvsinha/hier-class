import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

torch.manual_seed(233)

class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
            self.vocab_size = config.vocab_size
            self.embedding_dim = config.embedding_dim
            self.position_size = config.position_size
            self.position_dim = config.position_dim

            self.word_input_size = config.word_input_size
            self.sent_input_size = config.sent_input_size
            self.word_LSTM_hidden_units = config.word_GRU_hidden_units
            self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units


    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, hidden = self.sent_LSTM(sent_features)
        return enc_output, hidden

    # def initHidden(self):
    #     weight = next(self.parameters()).data
    #     return Variable(weight.new(self.n_layers, 1, self.hidden_size).zero_())


class DecoderRNN(nn.Module):
    def __init__(self, config):
        super(DecoderRNN, self).__init__()
        pass

class BLSTM(nn.Module):
    def __init__(self, config):
        super(BLSTM, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim

        self.num_classes = config.num_classes
        self.class_embedding_dim = config.class_embedding_dim

        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units


        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)
        self.class_embedding = nn.Embedding(self.num_classes, self.class_embedding_dim)
        # maybe here we can copy the graph subject embedding here
        # self.class_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))


        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)


        # self.encoder = nn.Sequential(nn.Linear(800, 400),
        #                              nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(400, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, self.num_classes),
                                     nn.Softmax())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)

        prob = self.decoder(enc_output)

        return prob.view(sequence_num, 1)
