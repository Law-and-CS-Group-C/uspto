import bz2
import pickle
import _pickle as cPickle
import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

from scipy import spatial

DO_NGRAMS = False
NGRAM_SIZE = 5
INPUT_WEEKS = 3
VECTOR_SIZE = 300
# EPOCHS = 40
EPOCHS = 5


def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

print("Loading input data...")
t0 = time.time()

data = []
i=0
for file in os.listdir('pickled'):
  print("opening {}".format(file))
  weekData = decompress_pickle('pickled/'+file)
  data += weekData
  # break
  i+=1
  if i==INPUT_WEEKS: break

# pikd = open('testdata.pickle', 'rb')
# data = pickle.load(pikd)
# pikd.close()

t1 = time.time()
print("{} patents loaded in {}s".format(len(data), t1-t0))



df = pd.DataFrame.from_records(filter(None,data))

# df.index = range(len(data))
# df['claims'].apply(len).sum() #count number of words

df = df.explode('claims', ignore_index=True) #create new row for each elem in each list of claims, copying the other col values
print("{} claims in loaded data".format(df.size))


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def cleanText2(text):
    text = re.sub(r'<[^>]*>', r' ', text)
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    # text = text.replace('x', '')
    return text

print("Cleaning text...")
t0 = time.time()
df['claims'] = df['claims'].apply(cleanText2)
t1 = time.time()
print("Cleaned in {}s".format(t1-t0))

if DO_NGRAMS:
    def splitNgram(text):
        tokens = [token for token in text.split(" ") if token != ""]
        ngramList = list(ngrams(tokens, NGRAM_SIZE))
        return [list(ngram) for ngram in ngramList]

    print("Splitting into ngrams...")
    df['claims'] = df['claims'].apply(splitNgram)
    df = df.explode('claims', ignore_index=True) #create new row for each elem in each ngram
    print("{} ngrams in loaded data".format(df.size))



# train, test = train_test_split(df, test_size=0.3, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def tokenize_text2(text):
    text = re.sub(r'[,\.;:\'\"]', r'', text)
    return text.lower().split(' ')


print("Converting to TaggedDocuments...")
t0 = time.time()

# train_tagged = train.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['claims']), tags=[r.patentNum]), axis=1)
# test_tagged = test.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['claims']), tags=[r.patentNum]), axis=1)

# train_tagged = train.apply(
#     lambda r: TaggedDocument(words=r['claims'], tags=[r.patentNum]), axis=1)
# test_tagged = test.apply(
#     lambda r: TaggedDocument(words=r['claims'], tags=[r.patentNum]), axis=1)


if DO_NGRAMS:
    # claims are already lists of words
    docs =  df.apply(
        lambda r: TaggedDocument(words=r['claims'], tags=[r.patentNum]), axis=1).values
else:
    docs =  df.apply(
        lambda r: TaggedDocument(words=tokenize_text2(r['claims']), tags=[r.patentNum]), axis=1).values

t1 = time.time()
print("Converted in {}s".format(t1-t0))

print("Building vocab...")
t0 = time.time()

import multiprocessing
cores = multiprocessing.cpu_count()

# model_dbow = Doc2Vec(dm=0, vector_size=VECTOR_SIZE, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
# model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

model_dbow = Doc2Vec(vector_size=VECTOR_SIZE, min_count=2, epochs=EPOCHS, workers=cores)
model_dbow.build_vocab(docs)

t1 = time.time()
print("Vocab built in {}s".format(t1-t0))

print("Training...")
t0 = time.time()

# model_dbow.train(docs, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

for epoch in range(EPOCHS):
    model_dbow.train(docs, total_examples=model_dbow.corpus_count, epochs=1)
    print("{}/{}".format(epoch+1,EPOCHS))

    # model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    # model_dbow.alpha -= 0.002
    # model_dbow.min_alpha = model_dbow.alpha

t1 = time.time()
print("Training done in {}s".format(t1-t0))

doc = docs[0]
# print(doc.tags)
vec = model_dbow.infer_vector(doc.words)
# print(vec)
# print(doc.words)


vecs = list(map(lambda doc : model_dbow.infer_vector(doc.words), docs))
vecs_dict = dict(zip(list(map(str,vecs)), docs.tolist()))
vecs = np.array(vecs)
tree = spatial.KDTree(vecs)

print("KDTree built")

def vectorize_claim(claim, patentNum):
    claim = cleanText(claim)
    doc = TaggedDocument(words=tokenize_text(claim), tags=[patentNum])
    return model_dbow.infer_vector(doc.words)

def kNearestClaims(claim_vec, tree, k):
    dists, indices = tree.query(claim_vec, k)
    similarities = list(map(lambda dist : 1/(1+dist), dists)) #converts vector distances into percentage similarities
    return similarities, indices

def findSimilarClaims(claim, patentNum):
    claim_vec = vectorize_claim(claim, patentNum)
    similarities, indices = kNearestClaims(claim_vec, tree, 3) #find 3 most similar claims
    for s, i in zip(similarities, indices):
        print("Similarity: ", s*100, "%")
        print("Claim: ", vecs_dict[str(vecs[i])].words)
        print('\n')
    #print(*(zip(similarities, indices)), sep='\n')

# print(df['claims'][0])
# print(df['patentNum'][0])
findSimilarClaims(df['claims'][0], df['patentNum'][0])




# print(vecs_dict[str(vecs[0])].words)
# print(tree.data[0]==vecs[0])
claim = ' '.join(vecs_dict[str(vecs[0])].words)
# print(claim)
# print(vectorize_claim(claim, df['patentNum'][0])==vectorize_claim(claim, df['patentNum'][0])) # this returns false so something stochastic is going on
# print(vecs[0]==vectorize_claim(claim, df['patentNum'][0]))



# linear search version to see if performance is better

def distBetween(vecA, vecB):
  d = 0.0
  # generalised pythagoras
  for axis in range(len(vecA)):
    axisDist = vecA[axis] - vecB[axis]
    d += axisDist*axisDist
  return d**0.5

def simBetween(vecA, vecB):
  # euclidean distance to cosine similarity
  similarity = 1 - spatial.distance.cosine(vecA, vecB)
  return similarity*100

def kNearestClaims2(claim_vec, k):
    distToEach = []
    for vec in vecs:
      distToEach.append([vec, simBetween(claim_vec, vec)])
    distToEach.sort(key=lambda x:x[1], reverse=True)
    return distToEach[:k]

def findSimilarClaimsLin(claim, patentNum):
    claim_vec = vectorize_claim(claim, patentNum)
    vecs = kNearestClaims2(claim_vec, 3) #find 3 most similar claims
    for vd in vecs:
        v = vd[0]
        d = vd[1]
        print("Similarity: ", d, "%")
        print("Claim: ", vecs_dict[str(v)].words)
        print('\n')

print(df['claims'][0])
print(df['patentNum'][0])
findSimilarClaimsLin(df['claims'][0], df['patentNum'][0])


print(simBetween(vectorize_claim(claim, df['patentNum'][0]), vectorize_claim(claim, df['patentNum'][0])))
print(distBetween(vectorize_claim(claim, df['patentNum'][0]), vectorize_claim(claim, df['patentNum'][0])))
#these numbers both change slightly on each run

