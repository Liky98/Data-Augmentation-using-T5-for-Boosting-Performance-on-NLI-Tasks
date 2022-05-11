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
    plt.plot(range(len(acc_list)), acc_list, label='Accuracy', color='darkred')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.imshow()
