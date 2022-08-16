import random

import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import transformers
from datasets import load_dataset,DatasetDict

from transformers import get_constant_schedule_with_warmup
from transformers import RobertaForSequenceClassification
from tqdm import tqdm
from transformers import AdamW
from tqdm.auto import tqdm
from datasets import load_metric
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def test(test_dataloader, model, device) :
    # test 데이터셋 성능 확인
    test_metric = load_metric("accuracy")
    model.to(device)
    prediction_list = []
    label_list = []
    model.eval() #모델 평가용도로 변경

    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        prediction_list.append(predictions)
        label_list.append(batch["labels"])
        test_metric.add_batch(predictions=predictions, references=batch["labels"])

    print(test_metric.compute())
    return prediction_list, label_list

def confusion(prediction_list, label_list) :
    # 혼동행렬
    my_data = []
    y_pred_list = []
    for data in prediction_list :
        for data2 in data :
            my_data.append(data2.item())
    for data in label_list :
        for data2 in data :
            y_pred_list.append(data2.item())

    confusion_matrix(my_data, y_pred_list)


    confusion_mx = pd.DataFrame(confusion_matrix(y_pred_list, my_data))
    ax =sns.heatmap(confusion_mx, annot=True, fmt='g')
    plt.title('confusion', fontsize=20)
    plt.show()

    print(f"precision : {precision_score(my_data, y_pred_list, average='macro')}")
    print(f"recall : {recall_score(my_data, y_pred_list, average='macro')}")
    print(f"f1 score : {f1_score(my_data, y_pred_list, average='macro')}")
    print(f"accuracy : {accuracy_score(my_data, y_pred_list)}")
    f1_score_detail= classification_report(my_data, y_pred_list,  digits=3)
    print(f1_score_detail)

def plot_accracy(acc_list) :
    plt.plot(range(1,len(acc_list)+1), acc_list, label='Accuracy', color='darkred')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def model_train(model_name, train_dataloader, dev_dataloader, device, num_epochs, num_label) :
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_label)
    model.to(device)
    # 옵티마이저 설정 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=1e-5)

    #num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * 2
        #num_training_steps= len(train_dataloader) * num_epochs
    )
    #lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    accuracy_list = []

    best_model_path = "BestModel.pth"
    final_path = best_model_path
    total_acc_train =0
    now_data_len =0

    train_dataloader = tqdm(train_dataloader, desc='Loading train dataset')
    for epoch in range(num_epochs):
        print(f"epoch : {epoch}")
        metric = load_metric("accuracy")
        losss = 0
        loss_count = 0

        for i, batch in enumerate(train_dataloader):
            model.train()

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss

            count_temp = 0
            for data in batch["labels"] :
                if data == 0 or data == 2 :
                    count_temp +=1
            
            
            # Neutral 관계에 대해 loss값 더줌
            if count_temp == 0 :
                loss = loss
            elif count_temp == 1:
                loss = loss*1.1
            elif count_temp == 2 :
                loss = loss*1.1*1.1
            elif count_temp == 3 :
                loss = loss * 1.1 * 1.1 * 1.1
            elif count_temp == 4 :
                loss = loss * 1.1 * 1.1 * 1.1 * 1.1
            

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            losss += loss.item()
            predict = torch.argmax(outputs.logits, dim=-1)

            acc = (predict == batch['labels']).sum().item()
            total_acc_train += acc
            now_data_len += len(batch['labels'])
            loss_count +=1

        # 성능확인
            train_dataloader.set_description(
                "Loss %.04f Acc %.04f | step %d Epoch %d" % (loss, total_acc_train / now_data_len, i, epoch))

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="모델 검증 중"):
                model.eval()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predict = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predict, references=batch["labels"])

        history = metric.compute()
        acc = history['accuracy']
        print(f"validation Accuracy => {acc}")
        print(f"Loss => {losss/loss_count}")
        accuracy_list.append(acc)

        if max(accuracy_list) <= acc :
            torch.save(model, final_path)


    return best_model_path, accuracy_list


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)  # type: ignore
  torch.backends.cudnn.deterministic = True  # type: ignore
  torch.backends.cudnn.benchmark = True  # type:


def snli_data_load(final_dataset_path):
    try :
        dataset = DatasetDict.load_from_disk(final_dataset_path)
        return dataset
    except :
        # 데이터 가져오기
        train_dataset_path = "../../Data/SNLI_train.csv"
        val_dataset_path = "../../Data/SNLI_dev.csv"
        test_dataset_path = "../../Data/SNLI_test.csv"


        data_files = {"train": train_dataset_path,
                      "validation": val_dataset_path,
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
        tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=4, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=4, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader, test_dataloader

if __name__ == "__main__" :
    set_seed(42)
    dataset = snli_data_load("./dataset")
    train_dataloader, eval_dataloader, test_dataloader = dataloader("roberta-base", dataset)
    best_model_path, accuracy_list = model_train("roberta-base",train_dataloader, eval_dataloader, torch.device("cuda:0"), 5, 3)
    best_model = torch.load(best_model_path)

    prediction_list, label_list = test(test_dataloader=test_dataloader,
                                            model=best_model,
                                            device=torch.device("cuda:0")
                                            )

    confusion(prediction_list=prediction_list,
                   label_list=label_list
                   )

    plot_accracy(accuracy_list)
