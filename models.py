import torch
import torch.nn as nn

class RNNModel(nn.ModuleList):
    def __init__(self,
                 num_layers=1,
                 hidden_size=256,
                 input_size=82,
                 model_type='rnn',
                 dropout=0.0,
                 vocab_len=2000,
                 ):
        super(RNNModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.model_type = model_type
        self.dropout = dropout
        self.vocab_len = vocab_len
        # self.L = self.num_layers * self.hidden_size

        self.emb_drop = nn.Dropout(p=dropout)
        # self.embedding = nn.Embedding(self.vocab_len, input_size)
        self.encoder = lambda xt: nn.functional.one_hot(xt.long(), input_size)

        if self.model_type == 'gru':
            self.rnn_layer = nn.GRU(input_size, self.hidden_size, batch_first=True,
                              bidirectional=False, dropout=self.dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size*3
        elif self.model_type == 'rnn':
            self.rnn_layer = nn.RNN(input_size, self.hidden_size, batch_first=True, nonlinearity='tanh',
                              bidirectional=False, dropout=self.dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size
        elif self.model_type == 'lstm':
            self.rnn_layer = nn.LSTM(input_size, self.hidden_size, batch_first=True,
                               bidirectional=False, dropout=self.dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size * 4

        self.rnn_drop = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_len)

    def forward(self, x, h):
        # print(x.shape)
        if x.shape[-1] == self.input_size:
            encoded = x
        else:
            encoded = self.encoder(x)
        # encoded = self.embedding(x.long())
        # print(encoded.shape)
        if self.model_type in ['lstm', 'rnn', 'gru']:
            self.rnn_layer.flatten_parameters()
        rnn_out, rnn_hn = self.rnn_layer(encoded.float(), h)
        # print('shapes', rnn_out.shape, rnn_hn.shape, h.shape)
        d_out = self.rnn_drop(rnn_out)
        y_hat = self.fc(d_out)
        return y_hat, rnn_out, rnn_hn

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if self.model_type == 'lstm':
            return (h, c)
        else:
            del c
            return h
