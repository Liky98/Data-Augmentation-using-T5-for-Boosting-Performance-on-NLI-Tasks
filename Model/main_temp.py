import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable
import LSTM
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import BertTokenizer

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--num_epochs", default=2000, type = int, help='size')
args.add_argument("--learning_rate", default=1e-5, type = int, help='size')
args.add_argument("--num_layers", default=1, type=int)
args.add_argument("--num_classes", default=1, type=int)

args.add_argument("--input_size", default=1, type = int, help='LSTM Model Input size')
args.add_argument("--hidden_size", default=2, type= int, help='LSTM')
args.add_argument("--dropout", default=0.3, type=float, help='LSTM')
args.add_argument("--seq_length", default=1, type=int, help='LSTM')
args = args.parse_args() #내용 저장

raw_data = ["Hello World", "Hello word"]
label = [0,1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_inp = tokenizer.encode_plus(raw_data, max_length=32, pad_to_max_length=True,
                                      return_tensors="pt")
tokenized_output = tokenizer.encode_plus(label, max_length=32, pad_to_max_length=True,
                                         return_tensors="pt")

print(tokenized_inp)

label = 1
#%%
from sentence_transformers import SentenceTransformer
transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

label = [0,1,2]

#Sentences are encoded by calling model.encode()
embeddings = transformer.encode(sentences)

input_batch = torch.tensor(sentences, dtype=torch.float32, requires_grad=True)
target_batch = torch.tensor(label, dtype=torch.int64)

print(input_batch)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
#%%
model = LSTM.LSTM(args)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)

for epoch in range(args.num_epochs) :
    for data in raw_data :
        #data = tokenizer.tokenize(data)
        data = transformer.encode(data)
        output = model(data)
        optimizer.zero_grad()

        loss = loss_function(output,label)

        loss.backward()
        optimizer.step()

        if epoch %100 == 0 :
            print("epoch: $d, loss : %1.5f" % (epoch, loss.item()))

#%%
sentences = ["i like dog", "i love coffee", "i hate milk", "you like cat", "you love milk", "you hate coffee"]
dtype = torch.float
word_list = list(set(" ".join(sentences).split()))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
print(word_dict)
print(number_dict)

batch_size = len(sentences)
n_step = 2  # 학습 하려고 하는 문장의 길이 - 1
n_hidden = 5  # 은닉층 사이즈

def make_batch(sentences):
  input_batch = []
  target_batch = []

  for sen in sentences:
    word = sen.split()
    input = [word_dict[n] for n in word[:-1]]
    target = word_dict[word[-1]]

    input_batch.append(np.eye(n_class)[input])  # One-Hot Encoding
    target_batch.append(target)

  return input_batch, target_batch

input_batch, target_batch = make_batch(sentences)

print(input_batch)
print(target_batch)
input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True)
target_batch = torch.tensor(target_batch, dtype=torch.int64)
print(input_batch)
print(target_batch)

