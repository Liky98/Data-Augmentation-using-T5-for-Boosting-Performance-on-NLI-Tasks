from Hugging모델모음 import Model_load
from Hugging_GLUE import GLUE_dataset

def dataset_name_list() :
    dataset = GLUE_dataset
    return dataset.Dataset_List(None)

def load_dataset_from_huggingface(dataset_name) :
    dataset = GLUE_dataset
    return dataset.load_GLUE_from_Huggingface(dataset_name)

def load_model_and_tokenizer() :
    model_class = Model_load
    model, tokenizer = model_class.albert_MaskedLM(None)
    return model, tokenizer

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

if __name__ == "__main__":
    dataset = load_dataset_from_huggingface("mnli")
    print(dataset)

    model, tokenizer = load_model_and_tokenizer()
    print(model)

    example = dataset["train"][:3]
    print(example)

    tokenized_datasets = example.map(tokenize_function, batched=True)
    tokenized_datasets
