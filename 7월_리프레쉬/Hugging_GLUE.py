from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import transformers
from datasets import load_dataset


class GLUE_dataset():
    def Dataset_List(self):
        print('cola', 'sst2', 'mrpc', 'qqp',
              'stsb', 'mnli', 'mnli_mismatched',
              'mnli_matched', 'qnli', 'rte', 'wnli', 'ax')


    def load_GLUE_from_Huggingface(name):
        print(f"GLUE-{name} dataset load")
        datasets = load_dataset("glue", name)
        return datasets


    def dataloader(model_name, dataset):
        # 토크나이저 설정
        tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

        def tokenize_function(example):
            return tokenizer(example["premise"], example["hypothesis"], truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # 데이터셋 이름 수정
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        # 데이터로더 정의
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=16, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"], batch_size=16, collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
        )

        return train_dataloader, eval_dataloader, test_dataloader



if __name__ == "__main__" :
    loadD = GLUE_dataset
    print(loadD.Dataset_List())
