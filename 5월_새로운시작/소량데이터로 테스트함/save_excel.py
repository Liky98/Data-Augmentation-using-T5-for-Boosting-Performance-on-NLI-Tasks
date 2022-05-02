import pandas as pd

def save_csv(augmentation_repository, file_name) :
    df = pd.DataFrame(augmentation_repository, columns=["premise","hypothesis"])
    df.to_csv('DA_{}.csv'.format(file_name))
    print(f" 데이터 증대 개수 : {len(df)}")
