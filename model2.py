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
    #the two lines below remove the importance scores of the keywords
    tfidf_dict = zip(*tfidf_dict)
    tfidf_dict = list(tfidf_dict)[0]
    return tfidf_dict[:NUM_KEYWORDS]

# returns: a tuple of the patent number,
# a string of relevant text (for the description),
# and the similarity score.
def findKNearestKeywordSet(dataset, query, k):
  return [{
      'id'    : patent[0],
      'text'  : ' '.join(patent[1]),
      'score' : random.random()
    }
    for patent in dataset[:k]]

patentKeywords = [
  (patent['patentNum'], extractKeywords('\n'.join(patent['claims'])))
  for patent in patentsById.values()
]

def findSimilarPatents(query, numResults):
  queryKeywords = extractKeywords(query)
  matches = findKNearestKeywordSet(patentKeywords, queryKeywords, numResults)

  results = []
  for match in matches:
    patent = patentsById[match['id']]
    date = patent['date']
    datePrettier = "{}-{}-{}".format(date[:4], date[4:6], date[6:])
    results.append({
      "patent": {
        "id": match['id'],
        "applicant": patent['applicant'],
        "date": datePrettier,
        "title": patent['title']
      },
      "relevantText" : match['text'],
      "similarity": match['score']
    })

  return results
