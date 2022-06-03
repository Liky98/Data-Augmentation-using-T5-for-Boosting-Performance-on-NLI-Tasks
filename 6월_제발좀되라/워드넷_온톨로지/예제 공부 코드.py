import nltk
"""

# Please use the NLTK Downloader to obtain the resource:
nltk.download('punkt')

# Please use the NLTK Downloader to obtain the resource:
nltk.download('averaged_perceptron_tagger')

# Please use the NLTK Downloader to obtain the resource:
nltk.download('wordnet')

"""
sentence = "Hello world."

tokens = nltk.word_tokenize(sentence)
print(tokens)

# (단어 토큰, 품사) 로 태깅됌
tagged = nltk.pos_tag(tokens)
print(tagged)

#%%
from nltk.corpus import wordnet

wordnet.synsets('shoot')    # 단어에 대한 동의어 집합
                            #('단어.품사.그룹인덱스') 형식으로 나옴.
#특정 의미의 단어를 보려면 '그룹인덱스'를 사용해서 명시적으로 지정을 해줘야함

#%%
app = wordnet.synset('apple.n.01')
print(app.definition()) # 표제어의 c단어 정의
print(app.lemma_names()) # 동의어 단어 집합


