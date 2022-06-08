"""
이 알고리즘은 Mihalcea et al. " Corpus-based and Knowledge-based Measures
of Text Semanticsimilarity" ( https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf )

"""
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def penn_to_wn(tag):
   #태그에서 대문자 나오면 소문자로 변경
    if tag.startswith('N'): #명사
        return 'n'

    if tag.startswith('V'): #동사
        return 'v'

    if tag.startswith('J'): #형용사
        return 'a'

    if tag.startswith('R'): #부사
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:         #바뀌는게 없으면 None 리턴
        return None

    try:
        return wn.synsets(word, wn_tag)[0]  #바뀌는게 있다면 동의어 있는지 확인해서 리턴
    except:
        return None #없다면


def sentence_similarity(sentence1, sentence2):
    sentence1 = pos_tag(word_tokenize(sentence1)) #토크나이저로 태깅하고
    sentence2 = pos_tag(word_tokenize(sentence2))

    #태그 확인하고 없으면 None, 있으면 n,v,a,r 에서 추가
    synsets1 = [tagged_to_synset(word, tag) for word, tag in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # None 있으면 지워버리기 ( Dropna 랑 같은거 )
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0


    for synset in synsets1: #문장1에 있는 단어 하나씩, 문장2에 모든 단어와 비교해서 가장 큰 값
        best_score = max([synset.wup_similarity(ss) for ss in synsets2])#path_similarity, lch_similarity, wup_similarity

        if best_score is not None: #만약 점수가 나오면
            score += best_score #점수 추가하고 카운트
            count += 1

    score /= count #점수를 단어 수로 나눠줌
    return score


def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2


dataset = pd.read_csv("../../Data/SNLI_dev.csv")

premise = dataset['premise']
hypothesis = dataset['hypothesis']
label = dataset['label']

score_list0 = []
score_list1 = []
score_list2 = []
for i in tqdm(range(len(premise))) :
    try:
        score = symmetric_sentence_similarity(premise[i],hypothesis[i])

        if label[i] == 0 :
            score_list0.append(score)
        elif label[i] == 1 :
            score_list1.append(score)
        else :
            score_list2.append(score)
    except:
        pass


print(f'label : 연관(0) , score : {np.mean(score_list0)}')
print(f'label : 모호(1) , score :{np.mean(score_list1)}')
print(f'label : 모순(2) , score : {np.mean(score_list2)}')


label = ["entailment", "neutral", "contradiction"]
plt.boxplot([score_list0, score_list1, score_list2])
plt.xlabel(label)
plt.legend()
plt.show()
