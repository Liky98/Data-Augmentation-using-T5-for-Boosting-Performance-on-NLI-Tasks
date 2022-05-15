from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
### 코사인 유사도 ###
def cos_performance(sentences) :

    tfidf_vectorizer = TfidfVectorizer()
     # 문장 벡터화(사전 만들기)

    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cos_similar[0][0]


### 유클리디언 유사도 (두 점 사이의 거리 구하기) ###
def euclidean_performance(sentences) :
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    ## 정규화 ##
    tfidf_normalized = tfidf_matrix/np.sum(tfidf_matrix)

    euc_d_norm = euclidean_distances(tfidf_normalized[0:1],tfidf_normalized[1:2])

    return euc_d_norm[0][0]

### 맨하탄 유사도(격자로 된 거리에서의 최단거리) ###
def manhattan_performance(sentences) :
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    ## 정규화 ##
    tfidf_normalized = tfidf_matrix/np.sum(tfidf_matrix)

    manhattan_d = manhattan_distances(tfidf_normalized[0:1],tfidf_normalized[1:2])

    return manhattan_d[0][0]

def sentence_transformer(doc_list,device) :
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

    # Tokenize sentences
    encoded_input = tokenizer(doc_list.to_list(), padding=True, truncation=True, max_length=200, return_tensors='pt')

    model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device)

    with torch.no_grad():
        model_output = model(input_ids=encoded_input["input_ids"].to("cuda"))

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'].to(device))

    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)
    return cosine_scores