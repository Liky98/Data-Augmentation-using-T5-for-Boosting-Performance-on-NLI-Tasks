from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import json
import torch
from transformers import AutoModelForSequenceClassification

# 경로 설정
snli_dataset_path = 'Data/SNLI/snli_1.0/snli_1.0/'
snli_train_path = snli_dataset_path + 'snli_1.0_train.jsonl'
snli_dev_path =  snli_dataset_path + 'snli_1.0_dev.jsonl'
snli_test_path = snli_dataset_path + 'snli_1.0_test.jsonl'

# 모델 선택
model_name = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
# 배치사이즈 설정
train_batch_size = 16


label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}# 연관, 모순, 중립

def _formatting(line):
    row = json.loads(line)
    x_1 = row['sentence1']
    x_2 = row['sentence2']
    y = label2int[row['gold_label']]
    return x_1, x_2, y

x_train_1, x_train_2, y_train = [], [], []      # Train 데이터 len() = 549,367
with open(snli_train_path, encoding='utf8') as f:
    for line in f:
        try:
            x_1, x_2, y = _formatting(line)
            x_train_1.append(x_1)
            x_train_2.append(x_2)
            y_train.append(y)
        except KeyError:
            continue

train_samples = []
for i in range(len(x_train_1)):
    train_samples.append(InputExample(texts=[x_train_1[i],x_train_2[i]], label=y_train[i]))

model_save_path = 'Classification/output/training_SNLI_roberta'
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples[:5000], batch_size=train_batch_size, name='SNLI-test')
test_evaluator(model, output_path=model_save_path)
model.eval()
print(test_evaluator(model, output_path=model_save_path))
#%%
print(model.eval())
model.evaluate()
