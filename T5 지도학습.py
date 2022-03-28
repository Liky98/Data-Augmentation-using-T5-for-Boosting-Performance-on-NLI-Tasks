"""
SNLI 데이터셋 여는 코드
"""
import json

SNLI_path = "Data/SNLI/snli_1.0/snli_1.0/"
train_path = SNLI_path + "snli_1.0_train.jsonl"

label2class = {'entailment': 0, 'contradiction': 1, 'neutral': 2} # 연관, 모순, 중립

def _formatting(line):
    row = json.loads(line)
    x_1 = row['sentence1']
    x_2 = row['sentence2']
    y = label2class[row['gold_label']]
    return x_1, x_2, y

x_train_1, x_train_2, y_train = [], [], []      # Train 데이터 len() = 549,367
with open(train_path, encoding='utf8') as f:
    for line in f:
        try:
            x_1, x_2, y = _formatting(line)
            x_train_1.append(x_1)
            x_train_2.append(x_2)
            y_train.append(y)
        except KeyError:
            continue

entailment_list = []
contradiction_list = []
neutral_list = []
for i in range(len(y_train)):
    if y_train[i] == 0 :
        entailment_list.append([x_train_1[i], x_train_2[i]])
    elif y_train[i] == 1 :
        contradiction_list.append([x_train_1[i], x_train_2[i]])
    else:
        neutral_list.append([x_train_1[i], x_train_2[i]])
#%% 데이터 수 확인하는 코드
print(f"연관된 데이터 수 : {len(entailment_list)}")
print(f"모순된 데이터 수 : {len(contradiction_list)}")
print(f"중립된 데이터 수 : {len(neutral_list)}")
""" 
연관된 데이터 수 : 183416
모순된 데이터 수 : 183187
중립된 데이터 수 : 182764
"""
#%% 모델 학습시키는 코드
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

max_source_length = 256
max_target_length = 256

# Examples 작성
input_sequence_1 = entailment_list[0][0]
output_sequence_1 = entailment_list[0][1]

input_sequence_2 = entailment_list[1][0]
output_sequence_2 = entailment_list[1][1]

# Tasl Description 작성
task_prefix = "entailment: "

# input 인코딩
input_sequences = [input_sequence_1, input_sequence_2]
encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequences],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# target 인코딩
target_encoding = tokenizer(
    [output_sequence_1, output_sequence_2], padding="longest", max_length=max_target_length, truncation=True
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

#%%
#예측
inputs = tokenizer("entailment: "+contradiction_list[0][0], return_tensors="pt", padding=True)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#%% 에측 확인
print(entailment_list[0][0])
print(entailment_list[0][1])
print(entailment_list[1][0])
print(entailment_list[1][1])
print("output => a person on a horse jumps over a broken down airplane.")

#%% temp

from transformers import AutoModelWithLMHead, AutoTokenizer

def get_question(Context, Hypothesis, max_length=128):
  input_text = "Context: %s  Hypothesis: %s </s>" % (Context, Hypothesis)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'],
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

context = "Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
answer = "Manuel"

get_question(answer, context)

# output: question: Who created the RuPERTa-base?