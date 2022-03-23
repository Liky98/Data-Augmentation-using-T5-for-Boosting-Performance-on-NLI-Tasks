"""
SNLI 데이터셋 여는 코드
"""
import json

SNLI_path = "SNLI/snli_1.0/snli_1.0/"
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

print(x_train_1[0])
print(x_train_2[0])
print(y_train[0])
print(len(x_train_1))
print(len(x_train_2))
print(len(y_train))
