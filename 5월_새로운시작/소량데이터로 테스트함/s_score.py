from tqdm import tqdm
import text_performance_indicators
from sentence_transformers import util
import save_excel
import pandas as pd
from datetime import datetime
def cos_simiraty(dataset) :
    processing_dataset = []
    for data in tqdm(dataset, desc="Dataset 점수 체크중 : "):
        data1 = text_performance_indicators.sentence_transformer(sentences=data)
        cosine_scores = util.pytorch_cos_sim(data1, data1)
        if data["label"] == 0 :
            if cosine_scores[0][1].item() >70 :
                processing_dataset.append(data)
        elif data["lable"] == 1 :
            if cosine_scores[0][1].item() >40 and cosine_scores[0][1].item() <70  :
                processing_dataset.append(data)
        elif data["lable"] == 2:
            if cosine_scores[0][1].item() < 40 :
                processing_dataset.append(data)

    df = pd.DataFrame(processing_dataset, columns=["premise", "hypothesis", "label"])
    df.to_csv('processing_{}.csv'.format(datetime.datetime.today().strftime("%D/%H:%M")), index=False)
    print(f" 데이터 증대 개수 : {len(df)}")