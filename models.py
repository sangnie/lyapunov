import torch
import torch.nn as nn

class RNNModel(nn.ModuleList):
    def __init__(self, mcon):
        super(RNNModel, self).__init__()

        self.num_layers = mcon.rnn_atts['num_layers']
        self.hidden_size = mcon.rnn_atts['hidden_size']
        self.model_type = mcon.model_type
        self.input_size = mcon.rnn_atts['input_size']
        dropout = mcon.dropout
        self.device = mcon.device

        # self.embedding = nn.Embedding(self.vocab_len, self.input_size)
        self.encoder = lambda xt: nn.functional.one_hot(xt.long(), self.input_size)

        if self.model_type == 'gru':
            self.rnn_layer = nn.GRU(self.input_size, self.hidden_size, batch_first=True,
                              bidirectional=False, dropout=dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size*3
        elif self.model_type == 'rnn':
            self.rnn_layer = nn.RNN(self.input_size, self.hidden_size, batch_first=True, nonlinearity='tanh',
                              bidirectional=False, dropout=dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size
        elif self.model_type == 'lstm':
            self.rnn_layer = nn.LSTM(self.input_size, self.hidden_size, batch_first=True,
                               bidirectional=False, dropout=dropout, num_layers=self.num_layers)
            # self.gate_size = self.hidden_size * 4

        self.dropout = nn.Dropout(p=dropout)
        # self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_len)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=mcon.output_size)

    def forward(self, x, h):
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
        d_out = self.dropout(rnn_out)
        y_hat = self.fc(d_out)
        return y_hat, rnn_out, rnn_hn

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        if self.model_type == 'lstm':
            return (h, c)
        else:
            del c
            return h
