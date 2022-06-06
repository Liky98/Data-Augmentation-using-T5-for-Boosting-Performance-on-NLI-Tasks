import nltk
from nltk.corpus import wordnet

sentences = [
    ["A snowboarder goes down a short hill next to stairs.", "A person is snowboarding."], # 연관관계
    ["A woman is talking on the phone while standing next to a dog.","A woman is walking her dog."] #모순관계
]

#토크나이저로 문장 자르기
for sentence in sentences :
    tokens = nltk.word_tokenize(sentence[0])
    print(tokens)
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    print()
    tokens = nltk.word_tokenize(sentence[1])
    print(tokens)
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    break

for sentence in sentences :
    sentence1 = sentence[0]
    sentence2 = sentence[1]

    tokens1 = nltk.word_tokenize(sentence1)
    print(tokens1)
    tagged = nltk.pos_tag(tokens1)
    print(tagged[0])

    x = wordnet.synsets(tagged[0][0])[0]
    print(x)

    print(x.lemma_names())

    break

#%%

# app = wordnet.synset('apple.n.01')
# print(app.definition()) # 표제어의 c단어 정의
# print(app.lemma_names()) # 동의어 단어 집합

#%%
hers = "herself"
print(wordnet.synsets(hers))
#print(wordnet.synset('dog.n.01').lemmas() )