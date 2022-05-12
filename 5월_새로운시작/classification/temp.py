import torch
import data_loader
import seed
import model_func
import test

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "save_model/DA_train_Nucleus 1 실험/BestModel.pth"

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

# model_path, accuracy_mean_list = model_func.model_train(model_name= model_name,
#                                               train_dataloader= train_dataloader,
#                                               dev_dataloader= validation_dataloader,
#                                               device= device,
#                                               save_path= model_save_path,
#                                               num_epochs = 15,
#                                               num_label = 3
#                                               )
best_model = torch.load(model_save_path)

prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= best_model,
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )

#test.plot_accracy(accuracy_mean_list)