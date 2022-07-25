from datasets import load_dataset
csv_file_path = '../../Data/SNLI_dev.csv'
dataset = load_dataset('csv', data_files={'train' : csv_file_path
                                          })
print(dataset)

print(dataset['train'][0])


from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

dataset['train'] = dataset['train'].filter(lambda x: x['label'] in [1, 2, 0])

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"])

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")

print(tokenized_datasets['train'][0])