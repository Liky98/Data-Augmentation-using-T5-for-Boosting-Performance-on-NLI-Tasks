import pandas as pd
import seed
import data_processing
import model_train
import torch
import Decoder
import save_excel
import s_score

seed.set_seed(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

entailment, neutral, contradiction = data_processing.data_load() #연관, 모호, 모순

# 연관 데이터셋 제작
entailment_file_name = "(0517)t5-base, trainData_100, entailment, beam_search, data_10000"

entailment_model = model_train.model_train(
    device=device,
    dataset= entailment[:100],
    epochs=5,
    path=entailment_file_name,
    set_max_length= 200,
    model_name='t5-base'
)
# 기훈이 화이팅 넌 할 수 있을꺼야 넌 인공지능 마스터니까 멋지다!!!!
entailment_outputs = Decoder.generation_sentence(model=entailment_model,
                                      dataset=entailment[1000: 11000],
                                      device=device,
                                      decoder_argorithm="beam_search",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(entailment_outputs, entailment_file_name,label=0)

# 모호 데이터셋 제작
neutral_file_name = "(0517)t5base, trainData_100, neutral, beam_search, data_10000"

neutral_model = model_train.model_train(
    device=device,
    dataset= neutral[:100],
    epochs=5,
    path=neutral_file_name,
    set_max_length= 200,
    model_name='t5-base'
)

neutral_outputs = Decoder.generation_sentence(model=neutral_model,
                                      dataset=neutral[1000: 11000],
                                      device=device,
                                      decoder_argorithm="beam_search",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(neutral_outputs, neutral_file_name,label=1)

# 모순 데이터셋 제작
contradiction_file_name = "(0517)t5base, trainData_100, contradiction, beam_search, data_10000"

contradiction_model = model_train.model_train(
    device=device,
    dataset= contradiction[:100],
    epochs=5,
    path=contradiction_file_name,
    set_max_length= 200,
    model_name='t5-base'
)

contradiction_outputs = Decoder.generation_sentence(model=contradiction_model,
                                      dataset=contradiction[1000: 11000],
                                      device=device,
                                      decoder_argorithm="beam_search",  #beam_search, nucleus_sampling
                                      model_name = 't5-base',
                                      setting_length= 200
                                      )

save_excel.save_csv(contradiction_outputs, contradiction_file_name,label=2)
#%%
#데이터셋 통합
data_save_path = "(Raw)t5base, trainData_100, beam_search, data_10000"

dataset_before = save_excel.integrated_csv(save_path = data_save_path,
                          contradiction_csv = 'DA_{}.csv'.format(contradiction_file_name),
                          entailment_csv = 'DA_{}.csv'.format(entailment_file_name),
                          neutral_csv = 'DA_{}.csv'.format(neutral_file_name)
                          )


#dataset_before = pd.read_csv("DA_(Raw)t5base, trainData_500, beam_search, data_10000.csv")
dataset_after = s_score.cos_simiraty(dataset_before,"0526")




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
""" beam_search > num_return_sequences=1
0506 15:16 통합 데이터셋 2만개 생성 
"""