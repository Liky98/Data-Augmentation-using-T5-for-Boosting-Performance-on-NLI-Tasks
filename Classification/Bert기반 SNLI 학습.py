import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
import torch.nn.functional as F
import json

"""
SNLI 데이터셋 여는 코드
"""

SNLI_path = "../Data/SNLI/snli_1.0/snli_1.0/"
train_path = SNLI_path + "snli_1.0_train.jsonl"

label2class = {'entailment': 0, 'contradiction': 1, 'neutral': 2} # 연관, 모순, 중립

def _formatting(line):
    row = json.loads(line)
    x_1 = row['sentence1']
    x_2 = row['sentence2']
    y = label2class[row['gold_label']]
    return x_1, x_2, y

x_train_1, x_train_2, y_train = [], [], []      # Train 데이터 len() = 549,367
with open(train_path, encoding='utf8') as f:
    for line in f:
        try:
            x_1, x_2, y = _formatting(line)
            x_train_1.append(x_1)
            x_train_2.append(x_2)
            y_train.append(y)
        except KeyError:
            continue


# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')
# train_data.head()

# train_df = train_data.dropna()
# test_df = test_data.dropna()
# train_df.shape #5,080 Rows 5 Columns

class load_dataset(Dataset) :
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = self.df.iloc[item, 3]
        label = self.df.iloc[item,4]
        return text, label

train_dataset = load_dataset(x_train_1,)
train_loader = DataLoader(train_dataset, shuffle=True)
device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-6)

""" 모델 학습 """
model.train()

total_corrct = 0
total_len = 0
total_loss = 0
count = 0

for epoch in range(1) :
    for text, label in train_loader:
        optimizer.zero_grad()

        encoding_list = [tokenizer.encode(x, add_special_tokens=True) for x in text] #한 문장에서 단어씩
                                                                                #<CLS>, <SEP> 등의 special token을 추가
        padding_list = [x + [0]*(256-len(x)) for x in encoding_list]

        sample = torch.tensor(padding_list)
        sample = sample.to(device)
        #label = torch.tensor(label)
        label = label.to(device)

        outputs = model(sample, labels=label)
        loss, logits = outputs
        print(logits)
        predict = torch.argmax(F.softmax(logits), dim=1)
        print(predict)
        corrct = predict.eq(label)

        total_corrct += corrct.sum()
        total_len += len(label)
        total_loss += loss

        loss.backward()
        optimizer.step()
        break
        if count % 1000 ==0 :
            print(f'Epoch : {epoch+1}, Iteration : {count}')
            print(f'Train Loss : {total_loss/1000}')
            print(f'Accuracy : {total_corrct/total_len}\n')

        count +=1

model.eval()
