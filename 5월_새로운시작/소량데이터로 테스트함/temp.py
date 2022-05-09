
import seed
import data_processing
import model_train
import torch
import Decoder
import save_excel

seed.set_seed(42)

# 연관 데이터셋 제작
contradiction_file_name = "Result/초기버전/(raw)T5_BASE, 1000 FewShot, beam_search 1, makeData 20000/contradiction.csv"
entailment_file_name ="Result/초기버전/(raw)T5_BASE, 1000 FewShot, beam_search 1, makeData 20000/entailment.csv"
neutral_file_name =  "Result/초기버전/(raw)T5_BASE, 1000 FewShot, beam_search 1, makeData 20000/DA_t5base, trainData_1000, neutral, beam_search1, data_20000.csv"

#데이터셋 통합
data_save_path = "beam_search 1 실험"

save_excel.integrated_csv(save_path = data_save_path,
                          contradiction_csv =contradiction_file_name,
                          entailment_csv = entailment_file_name,
                          neutral_csv =neutral_file_name
                          )

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

#%%
a = "aaa"
b = "bbb"
c = a+b
print(c)
