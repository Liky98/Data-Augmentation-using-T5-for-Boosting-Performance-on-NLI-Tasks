from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import transformers
from datasets import load_dataset,DatasetDict
from tqdm import tqdm

ds = load_dataset('super_glue', 'multirc')
ds.save_to_disk('tempds')

ds = DatasetDict.load_from_disk('tempds')
def snli_data_load(final_dataset_path, integrated_csv_path):
    try :
        dataset = DatasetDict.load_from_disk(final_dataset_path)
        return dataset
    except :
        # 데이터 가져오기
        train_dataset_path = "../../Data/SNLI/SNLI_train.csv"
        val_dataset_path = "../../Data/SNLI/SNLI_dev.csv"
        test_dataset_path = "../../Data/SNLI/SNLI_test.csv"
        DA_dataset_path = integrated_csv_path

        data_files = {"train": train_dataset_path,
                      "validation": val_dataset_path,
                      "test": test_dataset_path}
        dataset = load_dataset("csv", data_files=data_files)
        da_dataset = DatasetDict.load_from_disk(DA_dataset_path)

        da_dataset.shuffle(seeds=42)

        dataset_size = round(da_dataset["train"].num_rows / 10)

        for size in tqdm(range(dataset_size * 8), desc="train 데이터셋에 80% 추가"):
            dataset["train"].add_item(da_dataset["train"][size])
        for size in tqdm(range(dataset_size*8, da_dataset["train"].num_rows), desc="val 데이터셋에 나머지 20%추가"):
            dataset["validation"].add_item(da_dataset["train"][size])


        dataset.shuffle(seeds=42)
        dataset.save_to_disk(final_dataset_path)
        return dataset


def dataloader(model_name, dataset) :
    #토크나이저 설정
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    #데이터셋 이름 수정
    tokenized_datasets  = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    #데이터로더 정의
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=32, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=32, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader, test_dataloader

if __name__ == "__main__" :
    dataset = snli_data_load()

    print(f"train dataset => {dataset['train'].num_rows}")
    print(f"validation dataset => {dataset['validation'].num_rows}")
    print(f"test dataset => {dataset['test'].num_rows}")

