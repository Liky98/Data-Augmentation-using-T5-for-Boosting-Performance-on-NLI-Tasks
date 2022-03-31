import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer
import csv
import re

test_data = pd.read_csv('test.csv')
test_df = test_data.fillna("null")
model = torch.load('predict_of_tweet_model.pth')

model.to(torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

test_dataset = test_df["text"] #2,158 Rows
processing_test_dataset = test_dataset.reset_index()["text"]
processing_test_dataset = DataLoader(processing_test_dataset)

final = []
for text in processing_test_dataset:
    encoding_list = [tokenizer.encode(x, add_special_tokens=True) for x in text]
    padding_list = [x + [0]*(256-len(x)) for x in encoding_list]

    sample = torch.tensor(padding_list)
    sample = sample.to(torch.device('cpu'))
    outputs = model(sample)
    predict = torch.argmax(outputs[0],dim=1)
    print(f'predict -> {predict}')
    final.append(predict.int())


re_pattern = re.compile(r'\d')
temp = final
target_list = list()

for x in temp :
    temp = re.findall(re_pattern, str(x))
    target_list.append(temp[0])

with open('predict.csv', 'w', newline='') as f :
    writer = csv.writer(f)
    writer.writerow(target_list)

data_id = pd.DataFrame(test_data['id'])
data_target = pd.DataFrame(target_list)
result = pd.concat([data_id, data_target], axis=1)
result.to_csv("result1.csv")
