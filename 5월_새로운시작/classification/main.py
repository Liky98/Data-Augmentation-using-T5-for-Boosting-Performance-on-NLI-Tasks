import torch
import data_loader
import seed
import model_func
import test

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "save_model/100Few_Beams/"

#저장될 최종 데이터셋 경로 설정
dataset_path = "dataset/100Few_Beams"

#증대 데이터만 합쳐논 DataDict path
da_train_csv_path = "../소량데이터로 테스트함/데이터셋/DA_val_(0517)t5base, trainData_500, nucleus_sampling, data_10000.csv"
da_all_csv_path = "../소량데이터로 테스트함/데이터셋/DA_(0517)t5base, trainData_500, nucleus_sampling, data_10000.csv"
dataset = data_loader.data_load_noRawDatset(final_dataset_path = dataset_path,
                                     da_train_csv_path=da_all_csv_path
                                     )

train_dataloader, validation_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                                  dataset=dataset
                                                                                  )


model_path, accuracy_mean_list = model_func.model_train(model_name= model_name,
                                              train_dataloader= train_dataloader,
                                              dev_dataloader= validation_dataloader,
                                              device= device,
                                              save_path= model_save_path,
                                              num_epochs = 10,
                                              num_label = 3
                                              )

best_model = torch.load(model_save_path+model_path)

prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= best_model,
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )

test.plot_accracy(accuracy_mean_list)