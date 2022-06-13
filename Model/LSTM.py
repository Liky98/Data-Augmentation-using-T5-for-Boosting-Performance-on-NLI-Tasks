import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--input_size", default=16, type = int, help='LSTM Model Input size')
args.add_argument("--hidden_size", default=16, type= int, help='LSTM')
args.add_argument("--dropout", default=0.3, type=float, help='LSTM')
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






