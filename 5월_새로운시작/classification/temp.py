import torch
import test
from tqdm import tqdm
import seed
import data_loader
from datasets import load_metric
path = "save_model/DA_train_Nucleus 1 실험.pth"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "save_model/DA_train_Nucleus 1 실험.pth"

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
history = []

model = torch.load(path)

metric = load_metric("accuracy")
#%% 성능확인
model.eval()
with torch.no_grad():
    for batch in tqdm(validation_dataloader, desc="모델 검증 중"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predict = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predict, references=batch["labels"])

        history.append(metric.compute())

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

