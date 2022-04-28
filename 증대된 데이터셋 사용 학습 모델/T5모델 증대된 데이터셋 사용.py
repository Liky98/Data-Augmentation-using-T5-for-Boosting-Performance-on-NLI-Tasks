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
import pandas as pd
#시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 학습 기록
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# 데이터 가져오기
train_dataset_path = "../Data/SNLI/SNLI_train.csv" #수정 해야함
val_dataset_path = "../Data/SNLI/SNLI_train.csv" #수정 해야함
test_dataset_path = "../Data/SNLI/SNLI_train.csv" #수정 해야함

DA_dataset_path = ""

#train validation test 불러오기 및 생성데이터셋 불러오기
train_dataset = pd.read_csv(train_dataset_path)
val_dataset = pd.read_csv(val_dataset_path)
test_dataset = pd.read_csv(test_dataset_path)
da_dataset = pd.read_csv(DA_dataset_path)
da_dataset = da_dataset.sample(frac=1).reset_index(drop=True) #행섞기

# 8:2 분할하여 추가
spilt_len = len(da_dataset) // 10
train_dataset.append(da_dataset[:spilt_len*8])
val_dataset.append(da_dataset[spilt_len*8:])

train_dataset.columns = ['premise', 'hypothesis', ' label']
val_dataset.columns = ['premise', 'hypothesis', ' label']
test_dataset.columns = ['premise', 'hypothesis', ' label']

#모델 저장 경로 설정
model_name = 'sentence-transformers/stsb-roberta-large'
model_save_path = 'output/training_roberta-large_STS_SNLI_0406.pth'

#토크나이저 설정
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_val = val_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)

#데이터셋 이름 수정
tokenized_datasets_train  = tokenized_datasets_train.remove_columns(["premise", "hypothesis"])
tokenized_datasets_train = tokenized_datasets_train.rename_column("label", "labels")
tokenized_datasets_train.set_format("torch")
tokenized_datasets_val  = tokenized_datasets_val.remove_columns(["premise", "hypothesis"])
tokenized_datasets_val = tokenized_datasets_val.rename_column("label", "labels")
tokenized_datasets_val.set_format("torch")
tokenized_datasets_test  = tokenized_datasets_test.remove_columns(["premise", "hypothesis"])
tokenized_datasets_test = tokenized_datasets_test.rename_column("label", "labels")
tokenized_datasets_test.set_format("torch")


#데이터 로더 정의
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets_train, shuffle=True, batch_size=32, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets_val, batch_size=32, collate_fn=data_collator
)
# 데이터 처리에 오류없는지 batch 확인, 전처리 끝
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

#
from transformers import RobertaForSequenceClassification
model =RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)


outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


# 옵티마이저 설정 및 스케줄러
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
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
#device = torch.device('cpu')
model.to(device)
device
# tqdm 라이브러리써서 진행률 표시
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))


# 평가 루프, 결과 확인하도록 메트릭스 설정
from datasets import load_metric
metric = load_metric("accuracy")

#학습시작
set_seed(42)

for epoch in range(num_epochs):
    metric = load_metric("accuracy")
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

# test 데이터셋 성능 확인
test_metric = load_metric("accuracy")
test_dataloader = DataLoader(
    tokenized_datasets_test, batch_size=32, collate_fn=data_collator
)
prediction_list = []
label_list = []
model.eval() #모델 평가용도로 변경
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
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
