from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

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