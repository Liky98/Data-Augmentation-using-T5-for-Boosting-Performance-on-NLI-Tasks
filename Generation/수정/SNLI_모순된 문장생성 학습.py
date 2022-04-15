import random
import numpy as np
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from datasets import load_dataset

""" 세팅 """
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
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
    if data['label']==0 :
        entailment.append([data['premise'], data['hypothesis']])
    if data['label'] == 1:
        neutral.append([data['premise'], data['hypothesis']])
    if data['label'] == 2:
        contradiction.append([data['premise'], data['hypothesis']])

# few-shot learning 하기위한 예제 입력
true_false_adjective_tuples = []
for i in range(len(contradiction[:100])):
    true_false_adjective_tuples.append((contradiction[i][0], contradiction[i][1]))

""" ㅇㅇ """
# optimizer 설정
# 모든 파라미터 w,r,t의 그레디언트 로스 계산하고 업데이트
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
# 쿠다로 설정
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t5_model.to(device)

# 모델 학습
t5_model.train()

epochs = 10

for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in true_false_adjective_tuples:
    input_sent = "implicate: "+input+ " </s>"
    ouput_sent = output+" </s>"

    tokenized_inp = tokenizer.encode_plus(input_sent,  max_length=96, pad_to_max_length=True,return_tensors="pt")
    tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=96, pad_to_max_length=True,return_tensors="pt")


    input_ids  = tokenized_inp["input_ids"]
    attention_mask = tokenized_inp["attention_mask"]

    labels= tokenized_output["input_ids"]
    decoder_attention_mask=  tokenized_output["attention_mask"]


    # forward 함수 -> decoder_input_ids 생성
    output = t5_model(input_ids=input_ids.to(device), labels=labels.to(device),
                      decoder_attention_mask=decoder_attention_mask.to(device),
                      attention_mask=attention_mask.to(device))
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#%%테스트
test_sentence = entailment # 연관된 데이터셋
test_sent = 'implicate: {} </s>'.format(test_sentence[0][0])
test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"]
test_attention_mask = test_tokenized["attention_mask"]

t5_model.eval()
beam_outputs = t5_model.generate(
    input_ids=test_input_ids.to(device),attention_mask=test_attention_mask.to(device),
    early_stopping=True,
    num_beams=10,
    num_return_sequences=3,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)

print(f"주어진 문장 : {test_sentence[0][0]}")
print(f"원래 문장 : {test_sentence[0][1]}")
print(f"만든 문장 : {sent}")

torch.save(t5_model,'entailment_220414.pth')