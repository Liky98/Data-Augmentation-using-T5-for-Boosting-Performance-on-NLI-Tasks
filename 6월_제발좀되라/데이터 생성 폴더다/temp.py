import random
from datasets import load_dataset

raw_datasets = load_dataset("snli")

raw_datasets['train'] = raw_datasets['train'].filter(lambda x: x['label'] in [1, 2, 0])
raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x: x['label'] in [1, 2, 0])
raw_datasets['test'] = raw_datasets['test'].filter(lambda x: x['label'] in [1, 2, 0])

print(raw_datasets['train'][:3])