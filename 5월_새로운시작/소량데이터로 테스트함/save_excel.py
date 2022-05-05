import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def save_csv(augmentation_repository, file_name) :
    df = pd.DataFrame(augmentation_repository, columns=["premise","hypothesis"])
    df.to_csv('DA_{}.csv'.format(file_name))
    print(f" 데이터 증대 개수 : {len(df)}")

def integrated_csv(save_path, contradiction_csv,entailment_csv,neutral_csv) :
    da_dataset = load_dataset('csv', data_files=contradiction_csv)
    entailment_da_dataset = load_dataset('csv', data_files=entailment_csv)
    neutral_da_dataset = load_dataset('csv', data_files=neutral_csv)

    for size in tqdm(range(da_dataset["train"].num_rows), desc="데이터셋 병합"):
        da_dataset["train"].add_item(entailment_da_dataset["train"][size])
        da_dataset["train"].add_item(neutral_da_dataset["train"][size])

    da_dataset.shuffle(seeds=42)
    da_dataset.save_to_disk(save_path)

    return da_dataset