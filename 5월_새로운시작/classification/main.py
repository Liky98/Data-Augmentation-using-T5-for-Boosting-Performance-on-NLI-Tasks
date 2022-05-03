import torch
import data_loader
import seed
import model
import test

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed.set_seed(42)
model_name= 'roberta-large'
model_save_path = "12만개 증대 데이터셋 추가 후 학습.pth"

train_dataset, val_dataset, test_dataset = data_loader.snli_data_load()

train_dataloader, eval_dataloader, test_dataloader = data_loader.dataloader(model_name= model_name,
                                                                            test_dataset=train_dataset,
                                                                            val_dataset=val_dataset,
                                                                            train_dataset=test_dataset
                                                                            )

model = model.model_train(model_name= model_name,
                          train_dataloader= train_dataloader,
                          dev_dataloader= eval_dataloader,
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