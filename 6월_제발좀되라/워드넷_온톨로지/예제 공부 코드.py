import nltk
from nltk.corpus import wordnet

""" 다운로드

# Please use the NLTK Downloader to obtain the resource:
nltk.download('punkt')

# Please use the NLTK Downloader to obtain the resource:
nltk.download('averaged_perceptron_tagger')

# Please use the NLTK Downloader to obtain the resource:
nltk.download('wordnet')

"""
sentence = "Hello world."

tokens = nltk.word_tokenize(sentence) #토큰화
print(tokens)

tagged = nltk.pos_tag(tokens)# (단어 토큰, 품사) 로 태깅
print(tagged)

#%%
wordnet.synsets('shoot')    #단어에 대한 동의어 집합
                            #('단어.품사.그룹인덱스') 형식으로 나옴.
#특정 의미의 단어를 보려면 '그룹인덱스'를 사용해서 명시적으로 지정을 해줘야함

#%%
app = wordnet.synset('apple.n.01')
print(app.definition()) # 표제어의 c단어 정의
print(app.lemma_names()) # 동의어 단어 집합


wordnet.synset.definition() # Synset의 단어 정의
wordnet.synset.lemmas() # Synset의 동의어들
wordnet.synset.lemma() #Synset의 반의어들
wordnet.synset.hypernyms() # Synset의 상위어들
wordnet.synset.hyponyms() # Synset의 하위어들
#%%
wordnet.synset('good.a.01').lemmas()[0]
wordnet.synset('good.a.01').lemmas()[0].antonyms()

#%%
wordnet.synset('good.a.01').lemmas()[0].name()
wordnet.synset('good.a.01').lemmas()[0].antonyms()[0].name()

#- synset[0].name( )하면 'plan.n.01'처럼 객체명(whole synset name)이 나옴.
#- synset[0].lemmas()[0].name()하면 'plan'으로 단어 원형 이름(Name for the lemma)이 딱 나옴.

#%%
could = 'car.n.01'
print(wordnet.synsets(could)[0].hypernyms())
a = wordnet.synsets('slowly')[0].name()    #단어에 대한 동의어 집합
print(wordnet.synset(a).lemmas()[0].antonyms())