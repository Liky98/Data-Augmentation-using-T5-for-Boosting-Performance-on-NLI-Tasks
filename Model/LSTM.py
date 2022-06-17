import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable

"""
nn.LSTM
input_size: input의 feature dimension을 넣어주어야 한다. time step이 아니라 feature dimension!
hidden_size: 내부에서 어떤 feature dimension으로 바꿔주고 싶은지를 넣어주면 된다.
num_layers: lstm layer를 얼마나 쌓을지
bias: bias term을 둘 것인가 (Default: True)
batch_first: batch가 0번 dimension으로 오게 하려면 이거 설정! 난 이거 설정 가정하고 설명했다. (Default: False)
dropout: 가지치기 얼마나 할지, generalization 잘안되면 이걸 조정하면 된다.
bidirectional: 양방향으로 할지 말지 (bidirectional 하면 [forward, backword] 로 feature dimension 2배 됨)

Input
input dimension은 (Batch, Time_step, Feature dimension) 순이다. (batch_first=True)

outputs는 (output, (hidden or hidden,cell)) 의 tuple 형태로 나오므로 주의해서 써야한다. 
(LSTM만 cell state있음) 대체로 첫번째 output만 쓴다. (task마다 다르지만)
"""

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

class simpleLSTM(nn.Module):
    def __init__(self, args):
        super(simpleLSTM, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.lstm = nn.LSTM(input_size=args.input_size, # 300,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_layers=args.num_layers, #1
                            batch_first=True,
                            bidirectional=True)
    def forward(self, x):
        output = self.lstm(x)
        return output


class LSTM_Jun(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_Jun, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.layer_1 = nn.Linear(hidden_size, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()  # Activation Func

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # Hidden State
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # Internal Process States
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)  # Reshaping the data for starting LSTM network
        out = self.relu(hn)  # pre-processing for first layer
        out = self.layer_1(out)  # first layer
        out = self.relu(out)  # activation func relu
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.layer_3(out)
        out = self.relu(out)
        out = self.layer_out(out)  # Output layer
        return out