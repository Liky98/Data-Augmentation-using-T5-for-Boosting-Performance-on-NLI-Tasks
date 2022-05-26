from tqdm import tqdm
import text_performance_indicators
from sentence_transformers import util
import save_excel
import pandas as pd
from datetime import datetime
def cos_simiraty(dataset, path) :
    processing_dataset = []

    for i in tqdm(range(len(dataset)), desc="Dataset 점수 체크중 : "):
        try:
            data1 = text_performance_indicators.sentence_transformer(sentences=[dataset["premise"][i],dataset["hypothesis"][i]])
            cosine_scores = util.pytorch_cos_sim(data1, data1)
            # print(dataset["label"][i])
            # print(cosine_scores[0][1].item())
            if dataset["label"][i] == 0 :
                if cosine_scores[0][1].item() >0.5622 and cosine_scores[0][1].item()<0.7709 :
                    processing_dataset.append([dataset["premise"][i],dataset["hypothesis"][i],dataset["label"][i]])
            elif dataset["label"][i] == 1 :
                if cosine_scores[0][1].item() >0.1931 and cosine_scores[0][1].item() < 0.4694  :
                    processing_dataset.append([dataset["premise"][i],dataset["hypothesis"][i],dataset["label"][i]])
            elif dataset["label"][i] == 2:
                if cosine_scores[0][1].item() >0.3970 and cosine_scores[0][1].item() < 0.6432 :
                    processing_dataset.append([dataset["premise"][i],dataset["hypothesis"][i],dataset["label"][i]])
        except:
            pass
    df = pd.DataFrame(processing_dataset, columns=["premise", "hypothesis", "label"])

    df.to_csv('{}processing.csv'.format(path), index=False)
    print(f" 데이터 증대 개수 : {len(df)}")

if __name__ == "__main__":

    dataset = "DA_(0517)t5base, trainData_500, nucleus_sampling, data_10000.csv"
    dataset = pd.read_csv(dataset)
    cos_simiraty(dataset)
