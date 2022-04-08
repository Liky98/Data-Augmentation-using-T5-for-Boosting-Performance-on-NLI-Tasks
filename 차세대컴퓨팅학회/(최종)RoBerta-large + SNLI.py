"""


"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import get_scheduler
import transformers
import random
import numpy as np

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
model_name = 'roberta-large'
model_save_path = 'output/training_roberta-large_SNLI_0405.pth'

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
# 데이터 처리에 오류없는지 batch 확인, 전처리 끝
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

#모델정의하고 모델에 배치 전달
from transformers import AutoModelForSequenceClassification
from transformers import RobertaForSequenceClassification
model =RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# 옵티마이저 설정 및 스케줄러
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# 쿠다로 설정
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# tqdm 라이브러리써서 진행률 표시
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
# 평가 루프, 결과 확인하도록 메트릭스 설정
from datasets import load_metric

metric = load_metric("accuracy")

#학습 시작
set_seed(42)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update()


    # 성능확인
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predict = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predict, references=batch["labels"])
    print(metric.compute())



torch.save(model,model_save_path)

#%% test 데이터셋 성능 확인

test_metric = load_metric("accuracy")
test_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
)
prediction_list =[]
label_list = []
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
#{'accuracy': 0.9129682410423453}
#{Trani Dataset -> 'accuracy': 0.967688266677831}

# 혼동행렬 및 모델 성능 확인
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