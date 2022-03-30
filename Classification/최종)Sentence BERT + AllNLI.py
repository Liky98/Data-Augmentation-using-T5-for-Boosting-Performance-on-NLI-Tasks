"""
BERT(or RoBERTA, DistilB 등의 트랜스 모델 선택가능)를 사용해 훈련
SNLI + MultiNLI(AllNLI) 데이터 세트에 ERT 등)
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

#### 디버깅 정보를 stdout에 출력하기 위한 코드 (없어도됌)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#### stdout에 디버깅 정보 print

#데이터셋이 존재하는지 확인. 없으면 다운로드하여 압축 해제
nli_dataset_path = 'data/AllNLI.tsv.gz'
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


#여기에는 bert-base-uncased, roberta-base, xlm-roverta-base 등 huggingface/트랜스포머 Pretrained Model 지정가능
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# 배치사이즈 설정
train_batch_size = 16

#모델 저장 경로 설정
model_save_path = 'output/training_AllNLI_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#huggingface/트랜스포머 모델(BERT, RobERTa, XLNet, XLM-R 등)을 사용하여 토큰을 임베딩에 매핑
word_embedding_model = models.Transformer(model_name)

# 평균 풀링을 적용하여 고정 크기의 문장 벡터 하나 얻기
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# AllNLI.tsv.gz 파일을 읽고 Training Dataset 생성
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            label_id = label2int[row['label']]
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))



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

##############################################################################
#
# 저장된 모델을 로드하고 STS 벤치마크 데이터 세트에 대한 성능을 평가
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
#%%
print(test_evaluator(model, output_path=model_save_path))
