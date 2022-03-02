# patent search attempt 2
# uses TF-IDF and sematch keyword comparison

INPUT_WEEKS = 3

NUM_KEYWORDS = 10 # how many keywords to take from TF-IDF

PREVIEW_MAX_WORDS = 50 # how long the search result text should be

import time
import bz2
import pickle
import _pickle as cPickle
import os
import time

import random
from sklearn.feature_extraction.text import TfidfVectorizer



def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

print("Loading input data...")
t0 = time.time()

patentsById = {}
i=0
for file in os.listdir('pickled'):
  print("opening {}".format(file))
  weekData = decompress_pickle('pickled/'+file)

  patentsById |= { patent['patentNum']:patent
    for patent in weekData
    if patent!=None
  }
  i+=1
  if i==INPUT_WEEKS: break


t1 = time.time()
print("{} patents loaded in {}s".format(len(patentsById), t1-t0))


### TF-IDF PRECALC
# (running the full tfidf on each search is too expensive O(n),
# so we precalc the tf-idf values of every word in the dataset
# then precalc the keywords of every patent
def tf(t,d):
  # count of t in d / number of words in d
  # t is a word, d is a list of words (to avoid repeated splitting)
  return d.count(t)/len(d)

import re
def strToWordList(s):
  s = re.sub(r'[;,\.:\(\)]', '',s)
  return s.replace('\n',' ').lower().split(' ')


#first calculate the doc freq of all words
# (ie how many docs each word appears in)
t0 = time.time()
from collections import Counter
df = Counter()
# df = {}
for patent in patentsById.values():
  # fullText = ' '.join(patent['claims']) + ' '+ patent['description']
  fullText = ' '.join(patent['claims'])
  wordList = strToWordList(fullText)
  wordSet = set(wordList)
  patent['wordList'] = wordList #to reuse later :)
  patent['wordSet'] = wordSet
  df.update(wordSet)

t1 = time.time()
print("df calced in {}s".format(t1-t0))
t0 = time.time()

# now make the idf of these terms
import math
numDocs = len(patentsById)
idfDict = {}
for word in df:
  idfDict[word] = math.log(numDocs/df[word])
del(df)
highestIdf = max(idfDict.values())
t1 = time.time()
print("tfidf prep done in {}s".format( t1-t0))

def tfIdf(t,d):
  # if t isn't in the dict, assume it has very high
  try:
    return tf(t,d) * idfDict[t]
  except KeyError:
    return tf(t,d) * highestIdf

# i=0
# for w in idfDict:
#   print(w,idfDict[w])
#   i+=1
#   if i>10:break

def extractKeywordsFast(wordList, wordSet):
  # scores = [(word, tfIdf(word, wordList)) for word in wordSet]
  length = len(wordList)
  scores = [(word, (wordList.count(word)/length) * idfDict[word]) for word in wordSet if word != ''] # function unfolding for speeeed

  scores.sort(key=lambda x:x[1])
  out = [w[0] for w in scores[-NUM_KEYWORDS:]]
  #print(out)
  return out

def extractKeywordsSafe(wordList):
  wordSet = set(wordList)
  scores = [(word, tfIdf(word, wordList)) for word in wordSet if word != '']
  scores.sort(key=lambda x:x[1])
  out = [w[0] for w in scores[-NUM_KEYWORDS:]]
  print(out)
  return out

t0 = time.time()
patentKeywords = [
  (patent['patentNum'], extractKeywordsFast(patent['wordList'], patent['wordSet']))
  for patent in patentsById.values()
]
t1 = time.time()
print("dataset keyword calc done in {}s".format( t1-t0))


#### END TF-IDF



# returns: a tuple of the patent number,
# a string of relevant text (for the description),
# and the similarity score.
def findKNearestKeywordSet(dataset, query, k):
  return [{
      'id'    : patent[0],
      'text'  : ' '.join(patent[1]), #TODO fix (this is the tfidf keywords in some random order)
      'score' : random.random()
    }
    for patent in dataset[:k]] #PLACEHOLDER


def findSimilarPatents(query, numResults):
  queryKeywords = extractKeywordsSafe(strToWordList(query))
  matches = findKNearestKeywordSet(patentKeywords, queryKeywords, numResults)

  results = []
  for match in matches:
    patent = patentsById[match['id']]

    fullText = (' '.join(patent['claims']) + ' ' + patent['description'])
    truncatedText = ' '.join(fullText.split(' ')[:PREVIEW_MAX_WORDS])

    date = patent['date']
    datePrettier = "{}-{}-{}".format(date[:4], date[4:6], date[6:])
    results.append({
      "patent": {
        "id": match['id'],
        "applicant": patent['applicant'],
        "date": datePrettier,
        "title": patent['title']
      },
      "relevantText" : truncatedText,
      "similarity": match['score']
    })

  return results
