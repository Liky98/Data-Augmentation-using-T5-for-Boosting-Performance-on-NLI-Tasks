import text_performance_indicators
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

file_path = "DA_train_Nucleus 1 실험.csv"
#file_path = "../../Data/SNLI/SNLI_train.csv"
file = pd.read_csv(file_path)
file.dropna()

#연관 모호 모순 / 0 1 2
entailment = []
neutral = []
contradiction = []

for i in tqdm(range(len(file)), desc="데이터 분류 중 ") :

    if file["label"][i] == 0:
        entailment.append([file["premise"][i], file["hypothesis"][i]])
    if file["label"][i] == 1:
        neutral.append([file["premise"][i], file["hypothesis"][i]])
    if file["label"][i] == 2:
        contradiction.append([file["premise"][i], file["hypothesis"][i]])

entailment_performance_cos = []
neutral_performance_cos = []
contradiction_performance_cos = []

entailment_performance_euclidean = []
neutral_performance_euclidean = []
contradiction_performance_euclidean = []

entailment_performance_manhattan = []
neutral_performance_manhattan = []
contradiction_performance_manhattan = []

for sentence in tqdm(entailment, desc="연관 데이터셋 코사인 유사도 측정") :
    try:
        entailment_performance_cos.append(text_performance_indicators.cos_performance(sentence))
    except :
        pass
for sentence in tqdm(neutral, desc="연관 데이터셋 코사인 유사도 측정")  :
    try:
        neutral_performance_cos.append(text_performance_indicators.cos_performance(sentence))
    except:
        pass
for sentence in tqdm(contradiction, desc="연관 데이터셋 코사인 유사도 측정")  :
    try:
        contradiction_performance_cos.append(text_performance_indicators.cos_performance(sentence))
    except:
        pass

for sentence in tqdm(entailment, desc="연관 데이터셋 유클리디언 유사도 측정") :
    try:
        entailment_performance_euclidean.append(text_performance_indicators.euclidean_performance(sentence))
    except:
        pass
for sentence in tqdm(neutral, desc="연관 데이터셋 유클리디언 유사도 측정")  :
    try:
        neutral_performance_euclidean.append(text_performance_indicators.euclidean_performance(sentence))
    except:
        pass

for sentence in tqdm(contradiction, desc="연관 데이터셋 유클리디언 유사도 측정")  :
    try:
        contradiction_performance_euclidean.append(text_performance_indicators.euclidean_performance(sentence))
    except:
        pass
for sentence in tqdm(entailment, desc="연관 데이터셋 맨하탄 유사도 측정") :
    try:
        entailment_performance_manhattan.append(text_performance_indicators.manhattan_performance(sentence))
    except:
        pass
for sentence in tqdm(neutral, desc="연관 데이터셋 맨하탄 유사도 측정")  :
    try:
        neutral_performance_manhattan.append(text_performance_indicators.manhattan_performance(sentence))
    except:
        pass
for sentence in tqdm(contradiction, desc="연관 데이터셋 맨하탄 유사도 측정")  :
    try:
        contradiction_performance_manhattan.append(text_performance_indicators.manhattan_performance(sentence))
    except:
        pass

print(file_path)
print()
print("연관 cos 평균")
print(np.mean(entailment_performance_cos))
print(np.std(entailment_performance_cos))
print("모호 cos 평균")
print(np.mean(neutral_performance_cos))
print("모순 cos 평균")
print(np.mean(contradiction_performance_cos))
print()

print("연관 유클리디언 평균")
print(np.mean(entailment_performance_euclidean))
print("모호 유클리디언 평균")
print(np.mean(neutral_performance_euclidean))
print("모순 유클리디언 평균")
print(np.mean(contradiction_performance_euclidean))
print()

print("연관 맨하탄 평균")
print(np.mean(entailment_performance_manhattan))
print("모호 맨하탄 평균")
print(np.mean(neutral_performance_manhattan))
print("모순 맨하탄 평균")
print(np.mean(contradiction_performance_manhattan))

#%%
import matplotlib.pyplot as plt
plt.title("All Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
#plt.title("Entailment cosine similarity")
plt.plot(entailment_performance_cos,'bo', markersize=0.5)
#plt.show()

#plt.title("neutral cosine similarity")
plt.plot(neutral_performance_cos,'yo', markersize=0.5)
#plt.show()

#plt.title("contradiction cosine similarity")
plt.plot(contradiction_performance_cos,'ro', markersize=0.5)
plt.show()


#%%
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
cos_score_list = 0
for data in entailment :
    data = text_performance_indicators.sentence_transformer(sentences= data, device=device)
    cos_score_list = cos_score_list + data

print(f'허깅페이스에서 가져온 센텐스트랜스포머 정확도 => {cos_score_list / len(entailment)}')


"""
data1 = text_performance_indicators.cos_performance(entailment[0])
data2 = text_performance_indicators.euclidean_performance(entailment[0])
data3 = text_performance_indicators.manhattan_performance(entailment[0])
print(entailment[0])
print(data1)
print(data2)
print(data3)
print()

data1 = text_performance_indicators.cos_performance(entailment[1])
data2 = text_performance_indicators.euclidean_performance(entailment[1])
data3 = text_performance_indicators.manhattan_performance(entailment[1])
print(entailment[1])
print(data1)
print(data2)
print(data3)
print()

data1 = text_performance_indicators.cos_performance(entailment[2])
data2 = text_performance_indicators.euclidean_performance(entailment[2])
data3 = text_performance_indicators.manhattan_performance(entailment[2])
print(entailment[2])
print(data1)
print(data2)
print(data3)
print()

"""