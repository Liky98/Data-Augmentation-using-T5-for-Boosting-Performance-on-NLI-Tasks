from transformers import AlbertTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset

def cToD(csv_file_path = '../../Data/SNLI_dev.csv') :
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    dataset = load_dataset('csv', data_files={'train' : csv_file_path
                                              })

    def split_train_test(dataset) :
        dataset= dataset['train']
        dataset = dataset.train_test_split(test_size=0.2)
        val_test = dataset['test'].train_test_split(test_size=0.5)

        return dataset,val_test

    dataset,val_test = split_train_test(dataset)
    print(f"Train 데이터 개수 : { dataset['train'].num_rows}")
    print(f"validation 데이터 개수 : {val_test['train'].num_rows}")
    print(f"test 데이터 개수 : {val_test['test'].num_rows}")

    dataset['train'] = dataset['train'].filter(lambda x: x['label'] in [1, 2, 0])
    val_test['train'] = val_test['train'].filter(lambda x: x['label'] in [1, 2, 0])
    val_test['test'] = val_test['test'].filter(lambda x: x['label'] in [1, 2, 0])

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_val_test = val_test.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    tokenized_val_test = tokenized_val_test.remove_columns(["premise", "hypothesis"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_val_test = tokenized_val_test.rename_column("label", "labels")

    tokenized_datasets.set_format("torch")
    tokenized_val_test.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        tokenized_val_test["train"], batch_size=4, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_val_test["test"], batch_size=4, collate_fn=data_collator
    )

    return train_dataloader, val_dataloader, test_dataloader

if __name__ =="__main__" :
    a,b,c = cToD()
    print(a)
    print(b)
    print(c)