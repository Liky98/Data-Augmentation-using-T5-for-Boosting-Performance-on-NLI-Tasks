import random
from datasets import load_dataset

def data_load() :
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

    random.shuffle(entailment)
    random.shuffle(neutral)
    random.shuffle(contradiction)

    return entailment, neutral, contradiction

if __name__ == "__main__" :
    entailment, neutral, contradiction = data_load()
    print("리스트에 잘 분할해서 저장되었는지 확인해보자")
    print(f"연관된 문장 예시 -> {entailment[100]}")
    print(f"모호한 문장 예시 -> {neutral[100]}")
    print(f"모순된 문장 예시 -> {contradiction[100]}")  # 잘 맞는지 확인
