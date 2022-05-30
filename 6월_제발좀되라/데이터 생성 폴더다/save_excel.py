import pandas as pd
import sklearn

def save_csv(augmentation_repository, file_name, label) :
    df = pd.DataFrame(augmentation_repository, columns=["premise","hypothesis"])
    df['label'] = label
    df.to_csv('{}.csv'.format(file_name), index=False)
    print(f" 데이터 증대 개수 : {len(df)}")

def integrated_csv(save_path, contradiction_csv,entailment_csv,neutral_csv) :
    dataFrame = pd.concat(
        map(pd.read_csv, [contradiction_csv, entailment_csv,neutral_csv]), ignore_index=False)
    da_dataset =sklearn.utils.shuffle(dataFrame)
    dataset_size = round(len(da_dataset) / 10) * 8
    da_dataset[:dataset_size].to_csv('Train_{}.csv'.format(save_path), index=False)
    da_dataset[dataset_size:].to_csv('Val_{}.csv'.format(save_path), index=False)
    da_dataset.to_csv('{}.csv'.format(save_path), index = False)
    return da_dataset

if __name__ == "__main__" :
    print("csv 반환 ")