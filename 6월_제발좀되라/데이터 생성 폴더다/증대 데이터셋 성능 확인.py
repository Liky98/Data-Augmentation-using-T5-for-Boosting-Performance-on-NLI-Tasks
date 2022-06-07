import text_performance_indicators
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

file_path = "../../Data/SNLI_dev.csv"
file = pd.read_csv(file_path)
file.dropna()

#연관 모호 모순 / 0 1 2
entailment = []
neutral = []
contradiction = []

entailment_performance_cos = []
neutral_performance_cos = []
contradiction_performance_cos = []

entailment_performance_euclidean = []
neutral_performance_euclidean = []
contradiction_performance_euclidean = []

entailment_performance_manhattan = []
neutral_performance_manhattan = []
contradiction_performance_manhattan = []


for i in tqdm(range(len(file)), desc="데이터 분류 중 ") :
    if file["label"][i] == 0:
        entailment.append([file["premise"][i], file["hypothesis"][i]])
    if file["label"][i] == 1:
        neutral.append([file["premise"][i], file["hypothesis"][i]])
    if file["label"][i] == 2:
        contradiction.append([file["premise"][i], file["hypothesis"][i]])


def cos_performance(dataset, dataset2,save_list) :
    for sentence in tqdm(dataset, desc="{} 데이터셋 코사인 유사도 측정".format(dataset2)):
        try:
            save_list.append(text_performance_indicators.cos_performance(sentence))
        except:
            pass

def euclidean_performance(dataset,dataset2, save_list) :
    for sentence in tqdm(dataset, desc="{} 데이터셋 유클리디언 유사도 측정".format(dataset2)) :
        try:
            save_list.append(text_performance_indicators.euclidean_performance(sentence))
        except:
            pass
def manhattan_performance(dataset,dataset2, save_list) :
    for sentence in tqdm(dataset, desc="{} 데이터셋 맨하탄 유사도 측정".format(dataset2)) :
        try:
            save_list.append(text_performance_indicators.manhattan_performance(sentence))
        except:
            pass

cos_performance(entailment,"entailment", entailment_performance_cos)
cos_performance(neutral,"neutral", neutral_performance_cos)
cos_performance(contradiction,"contradiction", contradiction_performance_cos)

euclidean_performance(entailment, "entailment",entailment_performance_euclidean)
euclidean_performance(neutral,"neutral", neutral_performance_euclidean)
euclidean_performance(contradiction, "contradiction",contradiction_performance_euclidean)

manhattan_performance(entailment,"entailment", entailment_performance_manhattan)
manhattan_performance(neutral,"neutral", neutral_performance_manhattan)
manhattan_performance(contradiction, "contradiction",contradiction_performance_manhattan)

print(file_path)

def print_means () :
    print("연관 cos 평균")
    print(np.mean(entailment_performance_cos))
    print("모호 cos 평균")
    print(np.mean(neutral_performance_cos))
    print("모순 cos 평균")
    print(np.mean(contradiction_performance_cos))

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

print(print_means())



plt.title("All Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(entailment_performance_cos,'bo', markersize=0.2, label='Entailment')
plt.plot(neutral_performance_cos,'yo', markersize=0.2, label='Neutral')
plt.plot(contradiction_performance_cos,'ro', markersize=0.2, label='Contradiction')
plt.legend()
plt.show()

label = ["entailment", "neutral", "contradiction"]
plt.boxplot([entailment_performance_cos,neutral_performance_cos,contradiction_performance_cos])
plt.xlabel(label)
plt.show()


#%%
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
entailment_cos_score_list = []
count = 0
for data in tqdm(entailment, desc= " 연관 데이터셋 점수 체크중 : "):
    data1 = text_performance_indicators.sentence_transformer(sentences= data)
    cosine_scores = util.pytorch_cos_sim(data1, data1)
    entailment_cos_score_list.append(cosine_scores[0][1].item())
    count = count+1
    # if count == 100:
    #     break
plt.title("entailment Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(entailment_cos_score_list, 'bo', markersize=5)
plt.show()

neutral_cos_score_list = []
count = 0
for data in tqdm(neutral, desc= " 모호 데이터셋 점수 체크중 : "):
    data1 = text_performance_indicators.sentence_transformer(sentences= data)
    cosine_scores = util.pytorch_cos_sim(data1, data1)
    neutral_cos_score_list.append(cosine_scores[0][1].item())
    count = count+1
    # if count == 100:
    #     break
plt.title("neutral Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(neutral_cos_score_list, 'yo', markersize=5)
plt.show()


contradiction_cos_score_list = []
count = 0
for data in tqdm(contradiction, desc= " 모호 데이터셋 점수 체크중 : "):
    data1 = text_performance_indicators.sentence_transformer(sentences= data)
    cosine_scores = util.pytorch_cos_sim(data1, data1)
    contradiction_cos_score_list.append(cosine_scores[0][1].item())
    count = count+1
    # if count == 100:
    #     break
plt.title("contradiction Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(contradiction_cos_score_list, 'ro', markersize=5)
plt.show()

#%%
plt.title("Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(entailment_cos_score_list, 'bo', markersize=5, label= "entailment")
plt.plot(neutral_cos_score_list, 'yo', markersize=5, label= "neutral")
plt.plot(contradiction_cos_score_list, 'ro', markersize=5, label= "contradiction")
plt.legend()
plt.show()
#%%
label = ["entailment", "neutral", "contradiction"]
plt.boxplot([entailment_cos_score_list,neutral_cos_score_list,contradiction_cos_score_list])
plt.xlabel(label)
plt.legend()

plt.show()
 # %%
print(file_path)
print("0:연관, 1:모호, 2:모순  // 평균")
print(np.mean(entailment_cos_score_list))
print(np.mean(neutral_cos_score_list))
print(np.mean(contradiction_cos_score_list))

#%%
Q1 = np.percentile(entailment_cos_score_list, 25)
Q2 = np.percentile(entailment_cos_score_list, 50)
Q3 = np.percentile(entailment_cos_score_list, 75)
temp = np.mean(entailment_cos_score_list)
print('연관된 데이터셋')
print(f'Q1 > {Q1}')
print(f'Q2 > {Q2}')
print(f'Q3 > {Q3}')
print(f'데이터의 평균 값 > {temp}')
print()

Q1 = np.percentile(neutral_cos_score_list, 25)
Q2 = np.percentile(neutral_cos_score_list, 50)
Q3 = np.percentile(neutral_cos_score_list, 75)
temp = np.mean(neutral_cos_score_list)
print('모호한 데이터셋')
print(f'Q1 > {Q1}')
print(f'Q2 > {Q2}')
print(f'Q3 > {Q3}')
print(f'데이터의 평균 값 > {temp}')
print()

Q1 = np.percentile(contradiction_cos_score_list, 25)
Q2 = np.percentile(contradiction_cos_score_list, 50)
Q3 = np.percentile(contradiction_cos_score_list, 75)
temp = np.mean(contradiction_cos_score_list)
print('모순된 데이터셋')
print(f'Q1 > {Q1}')
print(f'Q2 > {Q2}')
print(f'Q3 > {Q3}')
print(f'데이터의 평균 값 > {temp}')
print()
