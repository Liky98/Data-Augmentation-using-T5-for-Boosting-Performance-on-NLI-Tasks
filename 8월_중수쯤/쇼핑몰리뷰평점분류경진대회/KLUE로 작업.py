
from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def train(train_dataloader,val_dataloader):
    model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    len_train_data = 22500
    len_val_data = 2500

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    top_val_loss = 100
    top_val_accuracy = 0
    train_loss_list =[]
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(1):
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        now_data_len = 0

        train_dataloader = tqdm(train_dataloader, desc='Loading train dataset')
        for i, batch in enumerate(train_dataloader):

            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}


            output = model(input_ids=batch["input_ids"],
                           token_type_ids=batch['token_type_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])

            #batch_loss = lossfunction(output.logits, batch['labels'])
            output.loss.backward()
            optimizer.step()

            predict = torch.argmax(output.logits, dim=-1)

            total_loss_train = total_loss_train + output.loss.item()
            acc = (predict == batch['labels']).sum().item()
            total_acc_train += acc
            now_data_len += len(batch['labels'])

            train_dataloader.set_description("Loss %.04f Acc %.04f | step %d Epoch %d" % (output.loss, total_acc_train / now_data_len, i,epoch))

        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                output = model(**batch)

                predict = torch.argmax(output.logits, dim=-1)

                acc = (predict == batch['labels']).sum().item()

                #batch_loss = lossfunction(output.logits, batch['labels'])

                total_loss_val += output.loss.item()

                total_acc_val += acc
            print()
            print(f'Epochs: {epoch + 1} \n'
                  f'| Train Loss: {total_loss_train / len_train_data: .3f} \n'
                  f'| Train Accuracy: {total_acc_train / len_train_data: .3f} \n'
                  f'| Val Loss: {total_loss_val /len_val_data: .3f} \n'
                  f'| Val Accuracy: {total_acc_val / len_val_data: .3f}')
            train_acc_list.append(total_acc_train / len_train_data)
            train_loss_list.append(total_loss_train / len_train_data)
            val_acc_list.append(total_acc_val / len_val_data)
            val_loss_list.append(total_loss_val /len_val_data)

        # if total_loss_val > top_val_loss and total_acc_val < top_val_accuracy :
        #
        #     break
        # if total_loss_val < top_val_loss:
        #     top_val_loss = total_loss_val
        # if total_acc_val > top_val_accuracy:
        #     top_val_accuracy = total_acc_val
        model_path = "./model" + str(epoch) + ".pth"
        model.save_pretrained(model_path)

    return model, train_acc_list, train_loss_list, val_acc_list, val_loss_list

def data() :
    data = pd.read_csv("./dataset/train.csv")
    data['target'].unique()  # 1,2,4,5 만 있음.

    raw_train = load_dataset('csv', data_files='./dataset/train.csv')
    raw_test = load_dataset('csv', data_files='./dataset/test.csv')
    train, valid = raw_train['train'].train_test_split(test_size=0.1).values()
    review_dataset = DatasetDict({'train': train, 'valid': valid, 'test': raw_test['train']})

    def tokenize_function(example):
        return tokenizer(example["reviews"], truncation=True)

    tokenized_datasets = review_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["id", "reviews"])
    tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("target", "labels")
    tokenized_datasets['valid'] = tokenized_datasets['valid'].rename_column("target", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_datasets["valid"], shuffle=True, batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=8, collate_fn=data_collator)

    return train_dataloader, val_dataloader, test_dataloader

def final_test(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    prediction_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = model(**batch.to(device))

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            prediction_list.extend(predictions.cpu().tolist())
    submission = pd.read_csv("dataset/sample_submission.csv")
    submission["target"] = prediction_list
    submission.to_csv("submission.csv",index=False)
    return prediction_list


def confusion(prediction_list) :
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

def result_graph(train_acc_list, train_loss_list, val_acc_list, val_loss_list):
    epochs = [x for x in range(len(train_loss_list))]
    print(epochs)
    plt.plot(epochs, train_loss_list, 'r', label='Training loss')
    epochs = [x for x in range(len(val_loss_list))]
    plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    epochs = [x for x in range(len(train_acc_list))]
    plt.plot(epochs, train_acc_list, 'r', label='Training Accuracy')
    epochs = [x for x in range(len(val_acc_list))]
    plt.plot(epochs, val_acc_list, 'b', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    train_dataloader, val_dataloader, test_dataloader = data()

    model, train_acc_list, train_loss_list, val_acc_list, val_loss_list = train(train_dataloader, val_dataloader)
    result_graph(train_acc_list, train_loss_list, val_acc_list, val_loss_list)
    predict_list, label_list = final_test(model, test_dataloader)


    # submission = pd.read_csv("dataset/sample_submission.csv")
    # submission["target"] = predict_list
    # submission.to_csv("submission.csv",index=False)
    # #%%
    # model = AutoModelForSequenceClassification.from_pretrained("model19.pth")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # prediction_list = []
    # label_list = []
    # model.eval()
    # with torch.no_grad():
    #     for batch in tqdm(test_dataloader):
    #         outputs = model(**batch.to(device))
    #
    #         logits = outputs.logits
    #         predictions = torch.argmax(logits, dim=-1)
    #         prediction_list.extend(predictions.cpu().tolist())
    # submission = pd.read_csv("dataset/sample_submission.csv")
    # submission["target"] = prediction_list
    # submission.to_csv("submission34.csv",index=False)
    #
    #
    #
