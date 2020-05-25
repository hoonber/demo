from gensim.models import word2vec
import logging

a = 0
b = 0
with open('./data/text8','r',encoding='utf-8') as file:
    line = file.read()
    for char in line:
        b+=1
        print(char,end='')
        if b-a == 100:
            a = b
            print('\n')
        if a == 5000:
            break

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./data/text8')
model = word2vec.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
model.save('./data/text82.model')
print(model['that'])

