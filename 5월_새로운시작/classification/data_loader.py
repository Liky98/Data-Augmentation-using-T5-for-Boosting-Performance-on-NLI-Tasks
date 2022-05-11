from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import transformers
from datasets import load_dataset,DatasetDict
import pandas as pd
import sklearn

def snli_data_load(final_dataset_path, da_train_csv_path, da_val_csv_path):
    try :
        dataset = DatasetDict.load_from_disk(final_dataset_path)
        return dataset
    except :
        # 데이터 가져오기
        train_dataset_path = "../../Data/SNLI/SNLI_train.csv"
        val_dataset_path = "../../Data/SNLI/SNLI_dev.csv"
        test_dataset_path = "../../Data/SNLI/SNLI_test.csv"

        dataFrame = pd.concat(
            map(pd.read_csv, [train_dataset_path, da_train_csv_path]), ignore_index=False)
        da_dataset = sklearn.utils.shuffle(dataFrame)
        da_dataset.to_csv('train_temp.csv', index=False)

        dataFrame = pd.concat(
            map(pd.read_csv, [val_dataset_path, da_val_csv_path]), ignore_index=False)
        da_dataset = sklearn.utils.shuffle(dataFrame)
        da_dataset.to_csv('val_temp.csv', index=False)

        data_files = {"train": './train_temp.csv',
                      "validation": './val_temp.csv',
                      "test": test_dataset_path}

        dataset = load_dataset("csv", data_files=data_files)

        dataset.save_to_disk(final_dataset_path)

        print(f"Train Dataset => {dataset['train'].num_rows}")
        print(f"Validation Dataset => {dataset['validation'].num_rows}")
        print(f"Test Dataset => {dataset['test'].num_rows}")

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

# if __name__ == "__main__" :
#     # # 저장될 최종 데이터셋 경로 설정
#     # dataset_path = "as"
#     #
#     # # 증대 데이터만 합쳐논 DataDict path
#     # ab = "../소량데이터로 테스트함/DA_train_Nucleus 2 실험.csv"
#     # cd = "../소량데이터로 테스트함/DA_val_Nucleus 2 실험.csv"
#     #
#     # dataset = snli_data_load(dataset_path,ab,cd)
#     #
#     # print(f"train dataset => {dataset['train'].num_rows}")
#     # print(f"validation dataset => {dataset['validation'].num_rows}")
#     # print(f"test dataset => {dataset['test'].num_rows}")
#
#     print("안녕")