import seed
import data_processing
import model_train
import torch
import Decoder
import save_excel

seed.set_seed(42)
file_name = "t5base, trainData_1000, contradiction, nucleus_sampling1, data_20000"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

entailment, neutral, contradiction = data_processing.data_load() #연관, 모호, 모순

model = model_train.model_train(
    device=device,
    dataset= contradiction[:1000],
    epochs=5,
    path=file_name,
    set_max_length= 200,
    model_name='t5-base'
)

outputs = Decoder.generation_sentence(model=model,
                                      dataset=contradiction[1000: 21000],
                                      device=device,
                                      decoder_argorithm="nucleus_sampling",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(outputs, file_name)

""" nucleus sampling > num_return_sequences=2
0502 18:09 연관 데이터셋 2만개 생성 돌림 
0502 21:23 모호 데이터셋 2만개 생성 돌림
0503 00:49 모순 데이터셋 2만개 생성 돌림
총 12만개 데이터 증대 완료
"""
"""nucleus sampling > num_return_sequences=1
0504 22:02 연관 데이터셋 2만개 생성 돌림 
0504 23:13 모호 데이터셋 2만개 생성 돌림 
0505 01:11 무관 데이터셋 2만개 생성 돌림 

"""