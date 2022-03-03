# patent search attempt 2
# uses TF-IDF and sematch keyword comparison

INPUT_WEEKS = 1

NUM_KEYWORDS = 10 # how many keywords to take from TF-IDF

PREVIEW_MAX_WORDS = 50 # how long the search result text should be

import time
import bz2
import pickle
import _pickle as cPickle
import os
import time



#### SEMATCH

import nltk
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('omw-1.4')

t0 = time.time()
from sematch.semantic.similarity import WordNetSimilarity
wns = WordNetSimilarity()
t1=time.time()
print("sematch dataset loaded in {}s".format( t1-t0))


def get_word_list_similarity(w1, w2):
  total_similarity = 0
  compared = 0
  # pairs = []
  for w in w1:

    best_word = None
    best_sim = 0

    # find best word pair
    for m in w2:
      sim = 1.0 if w==m else wns.word_similarity(w, m, 'lin')

      if sim > best_sim:
        best_sim = sim
        best_word = m

    if best_word != None:
      # pairs.append((w, best_word, best_sim))
      total_similarity += best_sim
      compared += 1

  return 0 if compared == 0 else total_similarity / compared




# returns: a tuple of the patent number,
# a string of relevant text (for the description),
# and the similarity score.
def findKNearestKeywordSet(dataset, queryKeywords, k):
  nearest = [(None, 0, None)] * k
  for patent in dataset:
    patentId = patent[0]
    keywords = patent[1]

    score = get_word_list_similarity(keywords, queryKeywords)
    if score<nearest[-1][1]:
      continue #not in the top k
    else:
      #insert into nearest, maintaining sort
      del(nearest[-1])
      insertAt = 0
      while insertAt < k-1 and score < nearest[insertAt][1]:
        insertAt +=1
      nearest.insert(insertAt, (patentId, score, keywords))

  return [{
      'id'    : match[0],
      'text'  : ' '.join(match[2]), #todo add explainable output
      'score' : match[1]
    }
    for match in nearest]

#### END SEMATCH




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
  s = re.sub(r'[;,\.:\(\)\r]', '',s)
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

# pre-load the dictionary so we can find similar keywords quickly
def readDict(dict_file):
    with open(dict_file,'r') as f:
        words=[]
        for line in f:
            word=line.strip()
            m=words.append(word)
        return words
dict = readDict("2of4brif_dict.txt")

# function to find k synonyms for some word using the dictionary and sematch similarity measure
def kSimilarWords(word, k):
    words = [(w, sim) for w in dict if (sim := 1.0 if w==word else wns.word_similarity(w, word, 'lin'))]
    words.sort(key=lambda x:x[1], reverse=True)
    words = list(zip(*words))[0]
    return words[:k]

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
    print(match['text'])

    similarKeywords = []
    for keyword in queryKeywords:
        synonyms = kSimilarWords(keyword, 10)
        similarKeywords.append({
          "patentKeyword" : keyword,
          "synonyms": synonyms
        })

  return results, similarKeywords

def tfidfSentence(query):
  wordList = strToWordList(query)
  scores = [(word, tfIdf(word, wordList)) for word in wordList if word != '']
  return scores
