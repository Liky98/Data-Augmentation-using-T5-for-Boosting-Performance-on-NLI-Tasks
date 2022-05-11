import torch
import test
from tqdm import tqdm
import seed
import data_loader
from datasets import load_metric
import model_func
path1 = "save_model/DA_train_Nucleus 1 실험0번째Epoch.pth"
path2 = "save_model/DA_train_Nucleus 1 실험2번째Epoch.pth"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'

#저장될 최종 데이터셋 경로 설정
dataset_path = "dataset/DA_train_Nucleus 1 실험"

#증대 데이터만 합쳐논 DataDict path
da_train_csv_path = "../소량데이터로 테스트함/DA_train_Nucleus 1 실험.csv"
da_val_csv_path = "../소량데이터로 테스트함/DA_val_Nucleus 1 실험.csv"

dataset = data_loader.snli_data_load(final_dataset_path = dataset_path,
                                     da_train_csv_path=da_train_csv_path,
                                     da_val_csv_path=da_val_csv_path)
train_dataloader, validation_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                                  dataset=dataset
                                                                                  )
# model, accuracy_mean_list = model_func.model_train(model_name= model_name,
#                                               train_dataloader= train_dataloader,
#                                               dev_dataloader= validation_dataloader,
#                                               device= device,
#                                               save_path= model_save_path,
#                                               num_epochs = 10,
#                                               num_label = 3
#                                               )
#%%
history = []

model = torch.load(path1)

metric = load_metric("accuracy")
#%% 성능확인
model.eval()
user_metric = []
with torch.no_grad():
    for batch in tqdm(validation_dataloader, desc="모델 검증 중"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predict = torch.argmax(logits, dim=-1)

        user_metric.append([predict, batch["labels"]])
        metric.add_batch(predictions=predict, references=batch["labels"])

data = metric.compute()
print(data)
#%%
data['accuracy']
#%%
print(user_metric)

correct=0
all=0
for data in user_metric:
    print(data.)
    break
#%%
for data in user_metric :
    for x, y in data :
        if x==y :
            correct = correct + 1
        all = all+1

print(f"정확도 > {(correct/all)*100}%")

#%%
import numpy
mean_history = []
for data in history:
    mean_history.append(data['accuracy'])

average = numpy.mean(mean_history)
print(average)

#print(history.__getitem__())
#print(history)
#history.append(metric.compute()['accuracy'])

# #%%
# prediction_list, label_list = test.test(test_dataloader=test_dataloader,
#                                         model= model,
#                                         device= device
#                                         )
#
# test.confusion(prediction_list=prediction_list,
#                label_list=label_list
#                )

#%%
prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= model,
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )
