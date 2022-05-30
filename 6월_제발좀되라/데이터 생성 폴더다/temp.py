import text_performance_indicators
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

test_data = [["hello world", " Hi World"],["hello world", "hello word"]]

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
entailment_cos_score_list = []
count = 0
for data in tqdm(test_data, desc= " 연관 데이터셋 점수 체크중 : "):
    data1 = text_performance_indicators.sentence_transformer(sentences= data)
    print(data1)
    print()
    print(data1[0])
    print()
    print(data1[1])
    cosine_scores = util.pytorch_cos_sim(data1, data1)
    print()
    print(cosine_scores)
    print()
    entailment_cos_score_list.append(cosine_scores[0][1].item())
    print(entailment_cos_score_list)


plt.title("entailment Dataset cosine similarity")
plt.xlabel("Data Index")
plt.ylabel("Score")
plt.plot(entailment_cos_score_list, 'bo', markersize=5)
plt.show()
