import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
#%%

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--num_epochs", default=2000, type = int, help='size')
args.add_argument("--learning_rate", default=1e-5, type = int, help='size')
args.add_argument("--num_layers", default=1, type=int)
args.add_argument("--num_classes", default=2, type=int) # Label 값

args.add_argument("--input_size", default=32, type = int, help='LSTM Model Input size')
args.add_argument("--hidden_size", default=3, type= int, help='LSTM')
args.add_argument("--dropout", default=0.3, type=float, help='LSTM')
args.add_argument("--seq_length", default=32, type=int, help='LSTM')
args.add_argument("--batch_size", default=3, type=int)
args.add_argument("--dimension", default=3, type=int)
args = args.parse_args() #내용 저장

class LSTM_Jun(nn.Module):
    def __init__(self, args):
        super(LSTM_Jun, self).__init__()
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length

        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, #단어의 개수 / 은닉층 사이즈 / 레이어 개수
                            num_layers=args.num_layers, batch_first=True)

        self.layer_1 = nn.Linear(args.hidden_size, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128,args.num_classes)
        self.relu = nn.ReLU()  # Activation Func

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Hidden State
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Internal Process States

        nn.Sequential(

            nn.ReLU(),
            nn.Linear(),
            nn.
        )
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

# Numpy array상태로는 학습이 불가능하므로, Torch Variable 형태로 변경(data/grad/grad_fn)
raw_data = ["Hello World", "Hello word"]
label = [0,1,0]


#%%
print(sentence_embeddings.shape)
#%%
train_x_tensor = torch.Tensor(sentence_embeddings)

train_y_tensor = torch.Tensor(label)

# train_x_tensor_final = torch.reshape(train_x_tensor, (train_x_tensor.shape[0], 1, train_x_tensor.shape[1]))
#
# train_y_tensor_final = torch.reshape(train_y_tensor, (train_y_tensor.shape[0], 1, train_y_tensor.shape[1]))

print(train_x_tensor.shape)


#%%
LSTM_Jun = LSTM_Jun(args)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(LSTM_Jun.parameters(), lr=args.learning_rate)

for epoch in range(args.num_epochs):
    for data in train_y_tensor :
        outputs = LSTM_Jun(data.unsqueeze(0))

        optimizer.zero_grad()

        loss = loss_function(outputs, train_y_tensor)

        loss.backward()

        optimizer.step()  # improve from loss = back propagation

        if epoch % 200 == 0:
            print("Epoch : %d, loss : %1.5f" % (epoch, loss.item()))

#%%
input_data = torch.randn(5,3,10)
print(input_data)
