import s_score

import pandas as pd
dataset_before = pd.read_csv("DA_(0517)t5base, trainData_100, nucleus_sampling, data_10000.csv")
dataset_before = dataset_before.dropna(axis=0)
dataset_after = s_score.cos_simiraty(dataset_before,path="실험2")


#첫번째꺼는 트레인데이터 500
#두번째꺼는 트레인데이터 100
