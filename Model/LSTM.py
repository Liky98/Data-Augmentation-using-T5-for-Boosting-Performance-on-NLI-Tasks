import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--input_size", default=16, type = int, help='LSTM Model Input size')
args.add_argument("--hidden_size", default=16, type= int, help='LSTM')
args.add_argument("--dropout", default=0.3, type=float, help='LSTM')
args.add_argument("--num_classes", default=1, type=int, help='LSTM')
args.add_argument("--num_layers", default=1, type=int, help='LSTM')
args.add_argument("--seq_length", default=128, type=int, help='LSTM')
args = args.parse_args() #내용 저장


n_input_size = 16
n_hidden_size = 8


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


class LSTM2(nn.Module):
    def __init__(self, args):
        super(LSTM2,self).__init__()

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

    def forward(self,x):
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

