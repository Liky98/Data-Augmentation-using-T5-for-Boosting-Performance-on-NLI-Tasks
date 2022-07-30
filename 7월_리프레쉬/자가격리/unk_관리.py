from datasets import load_dataset
from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer
from tqdm import tqdm
import torch
#%%
tokenizer= AlbertTokenizer.from_pretrained('albert-base-v2')
raw_datasets = load_dataset("snli")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%
unk_list = 0
unk_list_example = []
for data in tqdm(range(raw_datasets['train'].num_rows)):
    sample = tokenizer.encode_plus(raw_datasets['train']['premise'][data])
    for i in sample.input_ids:
        if i == 1:
            unk_list += 1
            unk_list_example.append(sample)
    sample = tokenizer.encode_plus(raw_datasets['train']['hypothesis'][data])
    for i in sample.input_ids:
        if i == 1:
            unk_list += 1
            unk_list_example.append(sample)
#%%
print(unk_list)
print(unk_list_example[0])
print(tokenizer.decode(unk_list_example[0].input_ids))

#%%
from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
print(model.parameters)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
# print(outputs)
# last_hidden_states = outputs.last_hidden_state


