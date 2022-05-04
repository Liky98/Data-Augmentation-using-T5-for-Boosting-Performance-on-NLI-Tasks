
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import transformers
import pandas as pd
from sklearn.utils import shuffle

from datasets import dataset_dict
def snli_data_load():
    # 데이터 가져오기
    train_dataset_path = "../../Data/SNLI/SNLI_train.csv"
    val_dataset_path = "../../Data/SNLI/SNLI_dev.csv"
    test_dataset_path = "../../Data/SNLI/SNLI_test.csv"
    DA_dataset_path = "../소량데이터로 테스트함/nucleus 20000 dataset/60000Dataset.csv"

    #train validation test 불러오기 및 생성데이터셋 불러오기
    train_dataset = pd.read_csv(train_dataset_path)
    val_dataset = pd.read_csv(val_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)
    da_dataset = pd.read_csv(DA_dataset_path)

    da_dataset = shuffle(da_dataset, random_state=42)

    dataset_size = round(len(da_dataset) // 10)
    train_dataset = pd.concat([train_dataset, da_dataset[ : dataset_size* 8]])
    val_dataset = pd.concat([val_dataset, da_dataset[dataset_size* 8 : ]])

    train_dataset = shuffle(train_dataset, random_state=42)
    val_dataset = shuffle(val_dataset, random_state=42)
    test_dataset = shuffle(test_dataset, random_state=42)

    train_dataset.columns = ['premise', 'hypothesis', ' label']
    val_dataset.columns = ['premise', 'hypothesis', ' label']
    test_dataset.columns = ['premise', 'hypothesis', ' label']

    #dataset_dict.from_csv
    train_dataset = train_dataset
    #val_dataset = val_dataset.tolist()
    #test_dataset = test_dataset.tolist()

    print(train_dataset)
    return train_dataset, val_dataset, test_dataset


def dataloader(model_name, train_dataset, val_dataset, test_dataset) :
    #토크나이저 설정
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)

    tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_datasets_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)

    #데이터셋 이름 수정
    tokenized_datasets_train  = tokenized_datasets_train.remove_columns(["premise", "hypothesis"])
    tokenized_datasets_train = tokenized_datasets_train.rename_column("label", "labels")
    tokenized_datasets_train.set_format("torch")

    tokenized_datasets_val  = tokenized_datasets_val.remove_columns(["premise", "hypothesis"])
    tokenized_datasets_val = tokenized_datasets_val.rename_column("label", "labels")
    tokenized_datasets_val.set_format("torch")

    tokenized_datasets_test  = tokenized_datasets_test.remove_columns(["premise", "hypothesis"])
    tokenized_datasets_test = tokenized_datasets_test.rename_column("label", "labels")
    tokenized_datasets_test.set_format("torch")

    # 데이터 로더 정의
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets_train, shuffle=True, batch_size=32, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets_val, batch_size=32, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets_test, batch_size=32, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader, test_dataloader

if __name__ == "__main__" :
    train, val, test = snli_data_load()
    print(f"train dataset => {len(train)}")
    print(f"val dataset => {len(val)}")
    print(f"test dataset => {len(test)}")

    dataloader(model_name="roberta-large",
               train_dataset=train,
               val_dataset=val,
               test_dataset=test)