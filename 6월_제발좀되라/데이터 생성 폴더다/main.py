import datetime

import pandas as pd
import seed
import data_processing
import model_train
import torch
import Decoder
import save_excel
import s_score
from datetime import date
seed.set_seed(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

entailment, neutral, contradiction = data_processing.data_load() #연관, 모호, 모순



# 연관 데이터셋 제작
entailment_file_name = "(0607)Few1000 연관"
neutral_file_name = "(0607)Few1000 모호"
contradiction_file_name = "(0607)Few1000 모순"
data_save_path = "(0607)Few1000"


entailment_model = model_train.model_train(
    device=device,
    dataset= entailment[:500],
    epochs=3,
    path="Models/" + entailment_file_name,
    set_max_length= 200,
    model_name='t5-base'
)

# 기훈이 화이팅 넌 할 수 있을꺼야 넌 인공지능 마스터니까 멋지다!!!!
entailment_outputs = Decoder.generation_sentence(model=entailment_model,
                                      dataset=entailment[5000: 15000],
                                      device=device,
                                      decoder_argorithm="nucleus_sampling",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(entailment_outputs, entailment_file_name,label=0)

# 모호 데이터셋 제작

neutral_model = model_train.model_train(
    device=device,
    dataset= neutral[:500],
    epochs=3,
    path="Models/" +neutral_file_name,
    set_max_length= 200,
    model_name='t5-base'
)

neutral_outputs = Decoder.generation_sentence(model=neutral_model,
                                      dataset=neutral[5000: 15000],
                                      device=device,
                                      decoder_argorithm="nucleus_sampling",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(neutral_outputs, neutral_file_name,label=1)

# 모순 데이터셋 제작

contradiction_model = model_train.model_train(
    device=device,
    dataset= contradiction[:500],
    epochs=3,
    path="Models/" +contradiction_file_name,
    set_max_length= 200,
    model_name='t5-base'
)

contradiction_outputs = Decoder.generation_sentence(model=contradiction_model,
                                      dataset=contradiction[5000: 15000],
                                      device=device,
                                      decoder_argorithm="nucleus_sampling",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(contradiction_outputs, contradiction_file_name,label=2)

#데이터셋 통합
dataset_before = save_excel.integrated_csv(save_path = data_save_path,
                          contradiction_csv = '{}.csv'.format(contradiction_file_name),
                          entailment_csv = '{}.csv'.format(entailment_file_name),
                          neutral_csv = '{}.csv'.format(neutral_file_name)
                          )

#dataset_before = pd.read_csv("DA_(Raw)t5base, trainData_500, beam_search, data_10000.csv")
#%%
dataset_after = s_score.cos_simiraty(dataset_before,"0607 new")
