from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset

def snli_dataset(tokenizer):
    raw_datasets = load_dataset("snli")

    print(f"Train 데이터 개수 : {raw_datasets['train'].num_rows}")
    print(f"validation 데이터 개수 : {raw_datasets['validation'].num_rows}")
    print(f"test 데이터 개수 : {raw_datasets['test'].num_rows}")

    raw_datasets['train'] = raw_datasets['train'].filter(lambda x: x['label'] in [1, 2, 0])
    raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x: x['label'] in [1, 2, 0])
    raw_datasets['test'] = raw_datasets['test'].filter(lambda x: x['label'] in [1, 2, 0])

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"]
                         )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=4, collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=4, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader, test_dataloader