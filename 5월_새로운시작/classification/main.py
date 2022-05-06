import torch
import data_loader
import seed
import model
import test

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "save_model/T5_BASE, 1000 FewShot, nucleus 1, makeData 20000.pth"
dataset_path = "dataset/(최종본)T5_BASE, 1000 FewShot, nucleus 1, makeData 20000"

integrated_csv_path = "../소량데이터로 테스트함/T5_BASE, 1000 FewShot, nucleus 1, makeData 20000"

dataset = data_loader.snli_data_load(final_dataset_path = dataset_path,
                                     integrated_csv_path= integrated_csv_path)

train_dataloader, validation_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                                  dataset=dataset
                                                                                  )

model = model.model_train(model_name= model_name,
                          train_dataloader= train_dataloader,
                          dev_dataloader= validation_dataloader,
                          device= device,
                          save_path= model_save_path,
                          num_label = 3,
                          num_epochs = 3
                          )

prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= model,
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )
