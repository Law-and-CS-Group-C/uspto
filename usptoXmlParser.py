import xml.etree.ElementTree as ET
import re
import bz2
import pickle
import _pickle as cPickle
import zipfile
import time
import os
import os.path

DELETE_FILES = True


# Pickle a file and then compress it into a file with extension
# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def compressed_pickle(title, data):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

# unused - unzipping to disk & deleting is faster
def get_zip_file_handler(path):
    patentzip = open(path, 'rb')
    with zipfile.ZipFile(patentzip) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield (zipinfo.filename, thefile)

def cleanXmlToString(strB):
  return re.sub(r'\n+', r'\n', strB.decode('utf-8')).strip()

def handlePatentXml(patentStr):
  root = ET.fromstring(patentStr)

  if root.tag == "sequence-cwu": #ignore DNA sequences
    return

  patentNum = root.find('us-bibliographic-data-grant').find('publication-reference').find('document-id').find('doc-number').text
  datePublished = root.find('us-bibliographic-data-grant').find('publication-reference').find('document-id').find('date').text
  patentTitle = root.find('us-bibliographic-data-grant').find('invention-title').text
  # print(patentNum)

  firstApplicant =  list(root.find('us-bibliographic-data-grant').find('us-parties').find('us-applicants').iter('us-applicant'))[0].find('addressbook')
  if firstApplicant.find('orgname') != None:
    applicantName = firstApplicant.find('orgname').text
  elif firstApplicant.find('first-name') != None:
    applicantName = "{}, {}".format(
      firstApplicant.find('last-name').text,
      firstApplicant.find('first-name').text
    )
  else:
    applicantName = firstApplicant.find('last-name').text


  # print(patentTitle, datePublished, applicantName)

  # DESCRIPTION

  # strip tables
  for table in root.find('description').iter('tables'):
    table.clear()

  # strip description of drawings
  for table in root.find('description').iter('description-of-drawings'):
    table.clear()

  description = cleanXmlToString(ET.tostring(root.find('description'), encoding='utf-8', method='text'))
  # print(description)

  # CLAIMS
  claims = []
  for claim in root.find('claims').findall('claim'):
    claims.append(cleanXmlToString(ET.tostring(claim, encoding='utf-8', method='text')))
  # print(claims)

  return {
    'patentNum' : patentNum,
    'description' : description,
    'claims' : claims,
    'applicant' : applicantName,
    'title' : patentTitle,
    'date' : datePublished,
  }

def handleXmlCollection(f):
  patents=[]
  patentfile=""
  for line in f:
    if '<?xml version' in line: #start of new file
      if patentfile != '':
        patents.append(handlePatentXml(patentfile))
      patentfile = line
    else:
      patentfile+=line
  return patents

def downloadZip(url):
  output = os.popen('wget -P downloads '+url)
  while True:
      line = output.readline()
      if line:
          print(line, end='')
      else:
          break
  output.close()

def extractXml(path):
  output = os.popen('unzip -d downloads '+path)
  while True:
      line = output.readline()
      if line:
          print(line, end='')
      else:
          break
  output.close()


def deleteFile(filename):
  if DELETE_FILES:
    try:
      os.remove(filename) # delete zip
    except:
      pass

with open('links.txt') as f:
  for url in f:
    filename = url[url.rfind('/')+1:].strip()
    xmlFilename = filename[:-3]+'xml'
    pickleFilename = filename[:-3]+'pbz2'

    if os.path.isfile('pickled/'+pickleFilename):
      print("{} already exists".format(pickleFilename))
      deleteFile('downloads/'+filename) # delete zip
      deleteFile('downloads/'+xmlFilename) # delete xml
      continue

    if os.path.isfile('downloads/'+filename):
      print("{} already exists".format(filename))
    else:
      print("downloading {}".format(url))
      downloadZip(url)

    if os.path.isfile('downloads/'+xmlFilename):
      print("{} already exists".format(xmlFilename))
    else:
      print("extracting {}".format(xmlFilename))
      extractXml('downloads/'+filename)

    print("processing {}".format(xmlFilename))
    patents = []
    with open('downloads/'+xmlFilename) as f:
      patents = handleXmlCollection(f)
    print(len(patents))
    print("writing {}.pbz2".format(filename[:-4]))
    compressed_pickle('pickled/'+filename[:-4], patents)

    deleteFile('downloads/'+filename) # delete zip
    deleteFile('downloads/'+xmlFilename) # delete xml



# TIME COMPARISON
# reading from decompressed xml:
# 4825 patents scanned in 24.2535662651062s, compressed/written in 30.679911851882935s
# (unzipping to disk takes like 2 secs with unzip shell cmd)
# reading from compressed xml:
# 4825 patents scanned in 139.35642075538635s, compressed/written in 30.41242027282715s
# so not worth it

