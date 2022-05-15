import torch
import data_loader
import seed
import model_func
import test

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "save_model/DA_train_Beam 1 실험(val제거)/BestModel.pth"

#저장될 최종 데이터셋 경로 설정
dataset_path = "dataset/DA_train_Beam 1 실험"

#증대 데이터만 합쳐논 DataDict path
da_train_csv_path = "../소량데이터로 테스트함/DA_train_beam_search 1 실험.csv"
da_val_csv_path = "../소량데이터로 테스트함/DA_val_beam_search 1 실험.csv"

dataset = data_loader.snli_data_load(final_dataset_path = dataset_path,
                                     da_train_csv_path=da_train_csv_path,
                                     da_val_csv_path=da_val_csv_path)

train_dataloader, validation_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                                  dataset=dataset
                                                                                  )


best_model = torch.load(model_save_path)

prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= best_model,
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )
