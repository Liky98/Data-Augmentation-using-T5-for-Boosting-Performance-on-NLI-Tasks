import torch
from torch import nn
import argparse
import numpy as np
from torch import optim
"""
 RNN 셀에 입력되는 텐서의 모양은 (batch_size, timesteps, input_dim)
 """
args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--num_epochs", default=2000, type = float, help='size')
args.add_argument("--dimension", default=3, type=float)
args.add_argument("--input_size", default=3, type=float)
args.add_argument("--num_layers", default=3, type=float)
args.add_argument("--hidden_size", default=10, type=float)
args.add_argument("--sequence_length", default=3, type=float)

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
    self.input_size = args.input_size

    self.rnn = nn.RNN(input_size=self.input_size,
                      hidden_size=self.hidden_size,
                      num_layers=self.num_layers,
                      batch_first=True)

    self.fc = nn.Sequential(
        nn.Linear(args.hidden_size * args.sequence_length, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)# 초기 hidden state 설정하기.

    print(h0)
    out, _hidden_state = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    print(out)
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out


model = VanillaRNN(args)
#(batch_size, timesteps, input_dim)
sample_data = torch.from_numpy(np.random.randn(10,3,3))
label = torch.from_numpy(np.random.randn(10,10))

train = torch.utils.data.TensorDataset(sample_data, label)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

input_size = 5
num_layers = 2
hidden_size = 8

criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:
    seq, target = data # 배치 데이터.
    out = model(seq)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,

    optimizer.zero_grad() #

    loss.backward() # loss가 최소가 되게하는
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))
