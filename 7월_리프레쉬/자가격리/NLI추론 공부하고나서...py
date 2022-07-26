import csv_to_datasetdict
from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import SNLI_load_and_processing
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch
import torch.nn.functional as F

class ALBERT_base_classification(nn.Module):
    def __init__(self):
        super(ALBERT_base_classification, self).__init__()
        self.AlBERT_base_config = AlbertConfig(hidden_size=768,
                                              num_attention_heads=12,
                                              intermediate_size=3072,
                                              id2label={"0": "Entailment",
                                                        "1": "Neutral",
                                                        "2": "Contradiction"}
                                              )

        self.model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", config = self.AlBERT_base_config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.model(input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        return output

    def get_tokenizer(self):
        tokenizer= AlbertTokenizer.from_pretrained('albert-base-v2')
        return tokenizer

    def save(self):
        torch.save(self.model, './model.pth')

    def load(self):
        self.model = torch.load('./model.pth')

def test():
    model = ALBERT_base_classification()
    tokenizer = model.get_tokenizer()

    inputs = tokenizer("Hello, my dog is cute.[SEP] my cat is cute.", return_tensors="pt")
    print(inputs)
    with torch.no_grad():
        logits = model(**inputs).logits
    print(logits)
    predicted_class_id = logits.argmax().item()
    print(model.AlBERT_base_config.id2label[predicted_class_id])

def train(model,train_dataloader,val_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    len_train_data = 550152
    len_val_data = 10000

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    lossfunction = nn.CrossEntropyLoss()
    top_val_loss = 100
    top_val_accuracy = 0
    for epoch in range(5):
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        now_data_len = 0

        train_dataloader = tqdm(train_dataloader, desc='Loading train dataset')
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch['input_ids'],batch['token_type_ids'],batch['attention_mask'], batch['labels'])

            batch_loss = lossfunction(output.logits, batch['labels'])
            output.loss.backward()
            optimizer.step()

            predict = torch.argmax(output.logits, dim=-1)

            total_loss_train = total_loss_train + batch_loss.item()
            acc = (predict == batch['labels']).sum().item()
            total_acc_train += acc
            now_data_len += len(batch['labels'])

            train_dataloader.set_description("Loss %.04f Acc %.04f | step %d Epoch %d" % (batch_loss, total_acc_train / now_data_len, i,epoch))

        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader):

                output = model(**batch.to(device))

                predict = torch.argmax(output.logits, dim=-1)

                acc = (predict == batch['labels']).sum().item()

                batch_loss = lossfunction(output.logits, batch['labels'])

                total_loss_val += batch_loss.item()

                total_acc_val += acc
            print()
            print(f'Epochs: {epoch + 1} \n'
                  f'| Train Loss: {total_loss_train / len_train_data: .3f} \n'
                  f'| Train Accuracy: {total_acc_train / len_train_data: .3f} \n'
                  f'| Val Loss: {total_loss_val /len_val_data: .3f} \n'
                  f'| Val Accuracy: {total_acc_val / len_val_data: .3f}')

        if total_loss_val > top_val_loss and total_acc_val < top_val_accuracy :
            break
        if total_loss_val < top_val_loss:
            top_val_loss = total_loss_val
        if total_acc_val > top_val_accuracy:
            top_val_accuracy = total_acc_val

        model.save()

    return model

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
            prediction_list.append(predictions)
            label_list.append(batch["labels"])

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


if __name__ == "__main__" :
    model = ALBERT_base_classification()
    tokenizer = model.get_tokenizer()

    train_dataloader, dev_dataloader, test_dataloader = SNLI_load_and_processing.snli_dataset(tokenizer)
    #train_dataloader, dev_dataloader, test_dataloader = csv_to_datasetdict.cToD()

    model = train(model=model, train_dataloader=train_dataloader, val_dataloader=dev_dataloader)

    predict_list, label_list = final_test(model, test_dataloader)
    confusion(predict_list,label_list)
    #print(AlbertConfig())

