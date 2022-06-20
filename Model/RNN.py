import torch
from torch import nn
import argparse
import numpy as np

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--num_epochs", default=2000, type = int, help='size')
args.add_argument("--dimension", default=3, type=int)
args = args.parse_args() #내용 저장

class RNN_model(nn.Module):
    def __init__(self, args):
        super(RNN_model, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings= args.vocab_length,
                                      embedding_dim=args.embedding_dim)
        self.h_now = None
        self.h_next = None
        self.w_now = None
        self.w_next = None

        self.mat_mul1 = np.matmul(self.h_now,self.w_now)
        self.mat_mul2 = np.matmul(self.h_next,self.w_next)

        self.bias = None

        self.tanh = nn.Tanh()

    def forward(self):



print()
class VanillaRNN(nn.Module):
  def __init__(self, args):
    super(VanillaRNN, self).__init__()
    self.hidden_size = args.hidden_size
    self.num_layers = args.num_layers

    self.rnn = nn.RNN(args.input_size, args.hidden_size, args.num_layers, batch_first=True)

    self.fc = nn.Sequential(
        nn.Linear(args.hidden_size * args.sequence_length, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out


model = VanillaRNN()

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8