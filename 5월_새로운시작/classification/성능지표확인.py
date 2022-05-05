import test
import data_loader
import torch
model_name= 'roberta-large'
model_save_path = "12만개 증대 데이터셋 추가 후 학습.pth"
dataset_path = "dataset/12만개 증대 데이터셋 추가"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

dataset = data_loader.snli_data_load(dataset_path)

train_dataloader, validation_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                                  dataset=dataset
                                                                                  )
prediction_list, label_list = test.test(test_dataloader=test_dataloader,
                                        model= torch.load(model_save_path),
                                        device= device
                                        )

test.confusion(prediction_list=prediction_list,
               label_list=label_list
               )
