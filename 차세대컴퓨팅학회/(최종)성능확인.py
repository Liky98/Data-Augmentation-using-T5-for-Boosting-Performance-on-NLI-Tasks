"""
모델 성능 평가 코드
"""
from datasets import load_metric
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import transformers
import random
import numpy as np
import torch

############변경하는 칸###########
model_name = 'sentence-transformers/stsb-roberta-base'  #토크나이저를 위한 Huggingface 모델 이름 설정
model = torch.load('output/(최종)training_roberta_STS_SNLI.pth') #저장된 path 경로
##############아래는 건드릴 필요없음 ##############

#시드 고정
def set_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 학습 기록
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# 데이터 가져오기
raw_datasets = load_dataset("snli")

raw_datasets['train'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['test'] = raw_datasets['test'].filter(lambda x : x['label'] in [1, 2, 0])

#모델 저장 경로 설정
#토크나이저 설정
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

#데이터셋 이름 수정
tokenized_datasets  = tokenized_datasets.remove_columns(["premise", "hypothesis"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")


#데이터 로더 정의
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=32, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=32, collate_fn=data_collator
)

test_metric = load_metric("accuracy")
test_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
)
prediction_list =[]
label_list = []
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.eval() #모델 평가용도로 변경
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    prediction_list.append(predictions)
    label_list.append(batch["labels"])

    test_metric.add_batch(predictions=predictions, references=batch["labels"])

print(test_metric.compute())

# 혼동행렬
import matplotlib.pyplot as plt
my_data = []
y_pred_list = []
for data in prediction_list :
    for data2 in data :
        my_data.append(data2.item())
for data in label_list :
    for data2 in data :
        y_pred_list.append(data2.item())

from sklearn.metrics import confusion_matrix
confusion_matrix(my_data, y_pred_list)

import pandas as pd
import seaborn as sns
confusion_mx = pd.DataFrame(confusion_matrix(y_pred_list, my_data))
ax =sns.heatmap(confusion_mx, annot=True, fmt='g')
plt.title('confusion', fontsize=20)
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
print(f"precision : {precision_score(my_data, y_pred_list, average='macro')}")
print(f"recall : {recall_score(my_data, y_pred_list, average='macro')}")
print(f"f1 score : {f1_score(my_data, y_pred_list, average='macro')}")
print(f"accuracy : {accuracy_score(my_data, y_pred_list)}")
f1_score_detail= classification_report(my_data, y_pred_list)
print(f1_score_detail)
