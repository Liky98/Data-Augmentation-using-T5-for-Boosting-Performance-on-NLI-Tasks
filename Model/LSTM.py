import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size= args.hidden_size, dropout= args.dropout)
        self.weight = nn.Parameter(torch.randn([args.hidden_size,args.input_size]))
        self.bias = nn.Parameter(torch.randn([args.input_size]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_and_cell, x):
        x = x.transpose(0,1)
        output, hidden = self.lstm(x, hidden_and_cell)
        output = output[-1] # 최종예측 히든레이어 선택
        model = torch.mm(output,self.weight) +self.bias
        return model


class LSTM_RNN기반(nn.Module):
    def __init__(self, args):
        super(LSTM_RNN기반,self).__init__()

        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length

        self.lstm = nn.LSTM(input_size=args.input_size,
                            hidden_size= args.hidden_size,
                            num_layers= args.num_layers,
                            dropout= args.dropout,
                            batch_first=True) # 미니배치를 앞으로 ex) [20,5] > [5,20]

        self.fc = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, x):
        hidden = Variable(torch.zeros(self.num_layers,   #hidden 초기값
                                x.size(0),
                                self.hidden_size
                                ))
        cell_state = Variable(torch.zeros(self.num_layers, #cell state 초기값
                        x.size(0),
                        self.hidden_size
                        ))

        output, (hidden_next,cell_state_next) = self.lstm(x,(hidden,cell_state))
        hidden_output = hidden_next.view(-1, self.hidden_size)
        out = self.fc(hidden_output)

        return out

class LSTM_classification(nn.Module):
    def __init__(self,args): #, dimension=128
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(args.seq_length, 300)
        self.dimension = args.dimension
        self.lstm = nn.LSTM(input_size=args.input_size, # 300,
                            hidden_size=args.dimension,
                            num_layers=args.num_layers, #1
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=args.dropout)

        self.fc = nn.Linear(2 * args.dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out