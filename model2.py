# patent search attempt 2
# uses TF-IDF and sematch keyword comparison

INPUT_WEEKS = 3

NUM_KEYWORDS = 10 # how many keywords to take from TF-IDF


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

# patents = []
# i=0
# for file in os.listdir('pickled'):
#   print("opening {}".format(file))
#   weekData = decompress_pickle('pickled/'+file)
#   patents += weekData
#   # break
#   i+=1
#   if i==INPUT_WEEKS: break

pikd = open('testdata.pickle', 'rb')
patents = pickle.load(pikd)
pikd.close()

t1 = time.time()
print("{} patents loaded in {}s".format(len(patents), t1-t0))


# def cleanText2(text):
#     text = re.sub(r'<[^>]*>', r' ', text)
#     text = re.sub(r'\|\|\|', r' ', text)
#     text = re.sub(r'http\S+', r'<URL>', text)
#     text = text.lower()
#     return text
#
# print("Cleaning text...")
# t0 = time.time()
# df['claims'] = df['claims'].apply(cleanText2)
# t1 = time.time()
# print("Cleaned in {}s".format(t1-t0))

def extractKeywords(str):
    #calling the TfidfVectorizer
    vectorize= TfidfVectorizer()
    #fitting the model and passing our sentences right away:
    claims = df['claims'].tolist()
    claims.insert(0, claim)
    response= vectorize.fit_transform(claims)
    dict_of_tokens={i[1]:i[0] for i in vectorize.vocabulary_.items()}
    row = response[0]
    tfidf_dict ={dict_of_tokens[column]:value for (column,value) in zip(row.indices,row.data)}
    #sort dict
    tfidf_dict = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    return tfidf_dict[:NUM_KEYWORDS]

# returns: a tuple of the patent number,
# a string of relevant text (for the description),
# and the similarity score.
def findKNearestKeywordSet(dataset, query, k):
  return [(patent[0], ' '.join(patent[1]), random.random())
    for patent in dataset[:k]]

patentKeywords = [
  (patent['patentNum'], extractKeywords('\n'.join(patent['claims'])))
  for patent in patents
]

def findSimilarPatents(query, numResults):
  queryKeywords = extractKeywords(query)
  matches = findKNearestKeywordSet(patentKeywords, queryKeywords, numResults)

  # TODO augment patents with date/applicant/name here
  return [{
    "patent": {
      "id": patent[0],
      "applicant": "APPLICANT APPLICANT",
      "date": "1970-01-01",
      "title": "NAME NAME NAME NAME"
    },
    "relevantText" : patent[1],
    "similarity": patent[2]
  }

  for patent in matches]
