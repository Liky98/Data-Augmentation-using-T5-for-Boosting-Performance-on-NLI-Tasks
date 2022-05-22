import s_score

import pandas as pd
dataset_before = pd.read_csv("DA_(0517)t5base, trainData_500, nucleus_sampling, data_10000.csv")
dataset_after = s_score.cos_simiraty(dataset_before)

