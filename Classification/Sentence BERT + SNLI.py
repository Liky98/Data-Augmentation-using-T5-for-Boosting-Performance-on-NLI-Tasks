"""
SBERT를 사용해 훈련함
SNLI 데이터셋 사용
1000번째 교육 단계(Iteration)마다 STS Benchmark Dataset에 따라 모델 평가

"""
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

#### 디버깅 정보를 stdout에 출력하기 위한 코드 (없어도됌)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

nli_dataset_path = '../Data/SNLI/snli_1.0/snli_1.0/snli_1.0_train.jsonl'
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# 배치사이즈 설정
train_batch_size = 16

#모델 저장 경로 설정
model_save_path = 'output/training_SNLI_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#huggingface/트랜스포머 모델(BERT, RobERTa, XLNet, XLM-R 등)을 사용하여 토큰을 임베딩에 매핑
word_embedding_model = models.Transformer(model_name)

# 평균 풀링을 적용하여 고정 크기의 문장 벡터 하나 얻기
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# SNLI 파일을 읽고 Training Dataset 생성
logging.info("Read SNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}# 연관, 모순, 중립

def _formatting(line):
    row = json.loads(line)
    x_1 = row['sentence1']
    x_2 = row['sentence2']
    y = label2int[row['gold_label']]
    return x_1, x_2, y

x_train_1, x_train_2, y_train = [], [], []      # Train 데이터 len() = 549,367
with open(nli_dataset_path, encoding='utf8') as f:
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


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


#STS benchmark Dataset 열고 학습 평가용으로 사용
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# 트레이닝 구성
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# 모델 학습
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

test_snli_dataset_path = '../Data/SNLI/snli_1.0/snli_1.0/snli_1.0_train.jsonl'
x_test_1, x_test_2, y_test = [], [], []
with open(test_snli_dataset_path, encoding='utf8') as f:
    for line in f:
        try:
            x_1, x_2, y = _formatting(line)
            x_test_1.append(x_1)
            x_test_2.append(x_2)
            y_test.append(y)
        except KeyError:
            continue

test_samples = []
for i in range(len(x_test_1)):
    test_samples.append(InputExample(texts=[x_test_1[i],x_test_2[i]], label=y_test[i]))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)

print(test_evaluator(model, output_path=model_save_path))