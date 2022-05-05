import save_excel
contradiction_file_name = "t5base, trainData_1000, contradiction, nucleus_sampling1, data_20000"
entailment_file_name = "t5base, trainData_1000, entailment, nucleus_sampling1, data_20000"
neutral_file_name = "t5base, trainData_1000, neutral, nucleus_sampling1, data_20000"

data_save_path = "T5_BASE, 1000 FewShot, nucleus 1, makeData 20000"

save_excel.integrated_csv(save_path = data_save_path,
                          contradiction_csv = 'DA_{}.csv'.format(contradiction_file_name),
                          entailment_csv = 'DA_{}.csv'.format(entailment_file_name),
                          neutral_csv = 'DA_{}.csv'.format(neutral_file_name)
                          )
