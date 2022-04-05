import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import get_scheduler
import pandas as pd

# 학습 기록
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# 데이터 가져오기
raw_datasets = load_dataset("snli")

raw_datasets['train'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['validation'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['test'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])

#모델 저장 경로 설정
model_name = 'roberta-base'
model_save_path = 'output/training_roberta2'

#토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    tokenized_datasets["train"], shuffle=True, batch_size=16, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=16, collate_fn=data_collator
)
# 데이터 처리에 오류없는지 batch 확인, 전처리 끝
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

#모델정의하고 모델에 배치 전달
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelWithLMHead
model =AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# 옵티마이저 설정 및 스케줄러
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

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

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

torch.save(model,model_save_path)
writer.flush()
writer.close()
#%%
# 평가 루프, 결과 확인하도록 메트릭스 설정
from datasets import load_metric

metric = load_metric("accuracy")
model.eval() #모델 보여주는건가
#%%
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
#%%
metric.compute()

# #%%99a297928c9238d191f04d079ac21aea01b44a9f
#
# #%%
# import wandb
#
# wandb.init(project="my-test-project", entity="liky98")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }
# wandb.log({"loss": loss})
#
# # Optional
# wandb.watch(model)
