"""
1차 속도 문제 개선 (시간복잡도 개선)
t5-large 모델 사용 증대
"""
import random
import numpy as np
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import pandas as pd
from datasets import load_dataset

""" 초기 세팅 """
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
set_seed(42)

tokenizer = T5Tokenizer.from_pretrained('t5-large')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')


""" 데이터 전처리 """
raw_datasets = load_dataset("snli")

raw_datasets['train'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x : x['label'] in [1, 2, 0])
raw_datasets['test'] = raw_datasets['test'].filter(lambda x : x['label'] in [1, 2, 0])

entailment =[] #0
neutral =[] #1
contradiction = [] #2
for data in raw_datasets['train'] :
    if data['label']==0 :
        entailment.append([data['premise'], data['hypothesis']])
    if data['label'] == 1:
        neutral.append([data['premise'], data['hypothesis']])
    if data['label'] == 2:
        contradiction.append([data['premise'], data['hypothesis']])

# 중복된 Premise Sentence 제거 및 무작위순서
# entailment = list(set([tuple(set(item)) for item in entailment]))
# neutral = list(set([tuple(set(item)) for item in neutral]))
# contradiction = list(set([tuple(set(item)) for item in contradiction]))

""" ㅇㅇ """
# optimizer 설정
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

# few-shot learning 하기위한 예제 입력
true_false_adjective_tuples = []
index_length = len(entailment)
repository = []

# 모델 학습 함수 작성
def model_train(device, dataset, epochs, path) :
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-large') # 업데이트마다 모델 새로
    t5_model.to(device)
    t5_model.train()

    true_false_adjective_tuples = []
    for data in dataset :
        true_false_adjective_tuples.append(
            (data[0], data[1])
        )

    for epoch in range(epochs):
        print(f"Epoch = {epoch}")
        for input, output in true_false_adjective_tuples :
            input_sent = "implicate: " + input + " </s>"
            ouput_sent = output + " </s>"

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
    for data in raw_dataset :
        test_sent = 'implicate: {} </s>'.format(data[0])

        test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

        test_input_ids = test_tokenized["input_ids"]
        test_attention_mask = test_tokenized["attention_mask"]

        beam_outputs = t5_model.generate(
            input_ids=test_input_ids.to(device), attention_mask=test_attention_mask.to(device),
            early_stopping=True,
            num_beams=10,
            num_return_sequences=3,
            no_repeat_ngram_size=2
        )

        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            repository.append((data[0], sent))  # 원래 문장, 생성된 문장

    return repository

k = 10

#for dataset in [entailment, contradiction, neutral] :
# 연관된 데이터셋 저장
augmentation_repository = []
model = model_train(device=device, dataset=entailment[:len(entailment) // k], epochs=5, path='entailment1')
output = generation_sentence(model=model, dataset=entailment[len(entailment) // k:], device=device)
augmentation_repository.append(output)
print(f"entailment 데이터 증대, 1차 작업 완료 ")
model = model_train(device=device, dataset=entailment[len(entailment) // k:], epochs=5, path='entailment2')
output = generation_sentence(model=model, dataset=entailment[:len(entailment) // k], device=device)
augmentation_repository.append(output)
print(f" entailment 데이터 증대, 2차 작업 완료 ")
# items = list(set([tuple(set(item)) for item in augmentation_repository]))
df = pd.DataFrame.from_records(augmentation_repository)
df.to_excel('DA_{}.xlsx'.format('entailment'))
print(f" 데이터 증대 개수 : {len(augmentation_repository)}")

# 모순된 데이터셋 저장
augmentation_repository = []
model = model_train(device=device, dataset=contradiction[:len(contradiction) // k], epochs=5, path='contradiction1')
output = generation_sentence(model=model, dataset=contradiction[len(contradiction) // k:], device=device)
augmentation_repository.append(output)
print(f"contradiction 데이터 증대, 1차 작업 완료 ")
model = model_train(device=device, dataset=contradiction[len(contradiction) // k:], epochs=5, path='contradiction2')
output = generation_sentence(model=model, dataset=contradiction[:len(contradiction) // k], device=device)
augmentation_repository.append(output)
print(f" contradiction 데이터 증대, 2차 작업 완료 ")
# items = list(set([tuple(set(item)) for item in augmentation_repository]))
df = pd.DataFrame.from_records(augmentation_repository)
df.to_excel('DA_{}.xlsx'.format('contradiction'))
print(f" 데이터 증대 개수 : {len(augmentation_repository)}")

# 모호된 데이터셋 저장
augmentation_repository = []
model = model_train(device=device, dataset=neutral[:len(neutral) // k], epochs=5, path='neutral1')
output = generation_sentence(model=model, dataset=neutral[len(neutral) // k:], device=device)
augmentation_repository.append(output)
print(f"neutral 데이터 증대, 1차 작업 완료 ")
model = model_train(device=device, dataset=neutral[len(neutral) // k:], epochs=5, path='neutral2')
output = generation_sentence(model=model, dataset=neutral[:len(neutral) // k], device=device)
augmentation_repository.append(output)
print(f" neutral 데이터 증대, 2차 작업 완료 ")
# items = list(set([tuple(set(item)) for item in augmentation_repository]))
df = pd.DataFrame.from_records(augmentation_repository)
df.to_excel('DA_{}.xlsx'.format('neutral'))
print(f" 데이터 증대 개수 : {len(augmentation_repository)}")
