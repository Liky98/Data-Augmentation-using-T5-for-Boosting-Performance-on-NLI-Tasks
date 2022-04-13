import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
set_seed(42)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

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

# few-shot learning 하기위한 예제 입력
true_false_adjective_tuples = [
                               ("The cat is alive","The cat is dead"),
                               ("The old woman is beautiful","The old woman is ugly"),
                               ("The purse is cheap","The purse is expensive"),
                               ("Her hair is curly","Her hair is straight"),
                               ("The bathroom is clean","The bathroom is dirty"),
                               ("The exam was easy","The exam was difficult"),
                               ("The house is big","The house is small"),
                               ("The house owner is good","The house owner is bad"),
                               ("The little kid is fat","The little kid is thin"),
                               ("She arrived early","She arrived late."),
                               ("John is very hardworking","John is very lazy"),
                               ("The fridge is empty","The fridge is full")

]

# 모델 학습
t5_model.train()

epochs = 10

for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in true_false_adjective_tuples:
    input_sent = "falsify: "+input+ " </s>"
    ouput_sent = output+" </s>"

    tokenized_inp = tokenizer.encode_plus(input_sent,  max_length=96, pad_to_max_length=True,return_tensors="pt")
    tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=96, pad_to_max_length=True,return_tensors="pt")


    input_ids  = tokenized_inp["input_ids"]
    attention_mask = tokenized_inp["attention_mask"]

    labels= tokenized_output["input_ids"]
    decoder_attention_mask=  tokenized_output["attention_mask"]


    # forward 함수 -> decoder_input_ids 생성
    output = t5_model(input_ids=input_ids, labels=labels,decoder_attention_mask=decoder_attention_mask,attention_mask=attention_mask)
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#테스트
test_sent = 'falsify: The sailor was happy and joyful. </s>'
test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"]
test_attention_mask = test_tokenized["attention_mask"]

t5_model.eval()
beam_outputs = t5_model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
    max_length=64,
    early_stopping=True,
    num_beams=10,
    num_return_sequences=3,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)

#%%
