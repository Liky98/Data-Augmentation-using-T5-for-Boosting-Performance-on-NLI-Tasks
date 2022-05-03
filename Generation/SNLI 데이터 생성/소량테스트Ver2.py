"""
아오
def 빼내고 테스트 해보는 코드
"""
import random
import time

import numpy as np
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

""" 초기 세팅 """
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)  # type: ignore
  torch.backends.cudnn.deterministic = True  # type: ignore
  torch.backends.cudnn.benchmark = True  # type:
set_seed(42)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

""" 데이터 전처리 """
raw_datasets = load_dataset("snli")

raw_datasets['train'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['test'] = raw_datasets['test'].filter(lambda x : x['label'] in [1, 2, 0])

entailment =[] #0
neutral =[] #1
contradiction = [] #2
for data in raw_datasets['train'] :
    if data['label'] == 0 :
        entailment.append([data['premise'], data['hypothesis']])
    if data['label'] == 1:
        neutral.append([data['premise'], data['hypothesis']])
    if data['label'] == 2:
        contradiction.append([data['premise'], data['hypothesis']])

print("리스트에 잘 분할해서 저장되었는지 확인해보자")
print(entailment[100])
print(neutral[100])
print(contradiction[100]) #잘 맞는지 확인

random.shuffle(entailment)
random.shuffle(neutral)
random.shuffle(contradiction)

print("섞었는데도 잘 저장되었는지 확인해보자")
print(entailment[100])
print(neutral[100])
print(contradiction[100]) #섞어도 잘맞지?

"""optimizer 설정 """
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

# GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t5_model.to(device)

# few-shot learning 하기위한 예제 입력s
#true_false_adjective_tuples = []
#index_length = len(entailment) #데이터셋의 크기
repository = []

# 모델 학습 함수 작성
def model_train(device, dataset, epochs, path) :
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-large') # 업데이트마다 모델 새로
    t5_model.to(device)
    t5_model.train()

    true_false_adjective_tuples = [(data[0], data[1]) for data in dataset]
    for epoch in tqdm(range(epochs), desc='모델 학습 중 : '):
        for input, output in true_false_adjective_tuples :
            input_sent =  "falsify: "+input +" </s>"
            ouput_sent = output +" </s>"

            tokenized_inp = tokenizer.encode_plus(input_sent, max_length=96, pad_to_max_length=True,
                                                  return_tensors="pt")
            tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=96, pad_to_max_length=True,
                                                     return_tensors="pt")

            input_ids = tokenized_inp["input_ids"]
            attention_mask = tokenized_inp["attention_mask"]

            labels = tokenized_output["input_ids"]
            decoder_attention_mask = tokenized_output["attention_mask"]

            # forward 함수 -> decoder_input_ids 생성
            output = t5_model(input_ids=input_ids.to(device), labels=labels.to(device),
                              decoder_attention_mask=decoder_attention_mask.to(device),
                              attention_mask=attention_mask.to(device))
            loss = output[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(t5_model, '{}.pth'.format(path))

    return t5_model

def generation_sentence(model, dataset, device) :
    g_model = model
    g_model.to(device)
    raw_dataset = dataset

    g_model.eval()

    repository = []

    for data in tqdm(raw_dataset,desc='데이터 증대 진행률 : ') :
        test_sent = 'falsify: {} </s>'.format(data[0])

        test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

        test_input_ids = test_tokenized["input_ids"]
        test_attention_mask = test_tokenized["attention_mask"]

        # beam_outputs = g_model.generate(
        #     input_ids=test_input_ids.to(device), attention_mask=test_attention_mask.to(device),
        #     early_stopping=True,
        #     num_beams=10, #  Beam Search 각 타임스텝에서 가장 가능성 있는 num_beams개의 시퀀스를 유지하고, 최종적으로 가장 확률이 높은 가설을 선택하는 방법
        #     num_return_sequences=3,
        #     no_repeat_ngram_size=2,
        #     max_length = 128
        # )

        beam_outputs = g_model.generate(
            input_ids = test_input_ids.to(device), attention_mask = test_attention_mask.to(device),
            do_sample=True,  # 샘플링 전략 사용
            max_length=96,  # 최대 디코딩 길이는 50
            top_k=50,  # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
            top_p=0.95,  # 누적 확률이 95%인 후보집합에서만 생성
            num_return_sequences=2,  # 1개의 결과를 디코딩해낸다
            early_stopping = True
        )

        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            repository.append([data[0], sent])  # 원래 문장, 생성된 문장
            break

    return repository

#for dataset in [entailment, contradiction, neutral] :

# 연관된 데이터셋 저장
augmentation_repository = []
model = model_train(device=device, dataset=contradiction[80:100], epochs=5, path='contradiction_소량테스트0524 Nucleus')


output = generation_sentence(model=model, dataset=contradiction[100:200], device=device)
augmentation_repository.append(output)
print(f"contradiction 데이터 증대, 1차 작업 완료 ")


#%%df = pd.DataFrame.from_records(np.array(augmentation_repository).T)
df = pd.DataFrame(augmentation_repository[0], columns=["premise","hypothesis"])
df.to_csv('DA_{}.csv'.format('contradiction_소량테스트 0502 Nucleus'))
print(f" 데이터 증대 개수 : {len(df)}")
