#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import re
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from numpy import dot
from numpy.linalg import norm
from operator import add

sc = SparkContext()
spark = SparkSession(sc)


# In[2]:


def buildArray(listOfIndices):
    returnVal = np.zeros(20000)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


d_corpus = sc.textFile(sys.argv[1], 1)
d_keyAndText = d_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWords = d_keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = d_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(lambda x, y: x + y)

# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words. 
topWords = allCounts.top(20000, lambda x: x[1])

# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through 20000
# 20000 is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(20000))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
# task 1 -- frequency position


# Frequency position:
print("The frequency position of the words 'applicant' is: ",
      dictionary.filter(lambda x: x[0] == 'applicant').map(lambda x: x[1]).first())
print("The frequency position of the words 'and' is: ",
      dictionary.filter(lambda x: x[0] == 'and').map(lambda x: x[1]).first())
print("The frequency position of the words 'attack' is: ",
      dictionary.filter(lambda x: x[0] == 'attack').map(lambda x: x[1]).first())
print("The frequency position of the words 'protein' is: ",
      dictionary.filter(lambda x: x[0] == 'protein').map(lambda x: x[1]).first())
print("The frequency position of the words 'car' is: ",
      dictionary.filter(lambda x: x[0] == 'car').map(lambda x: x[1]).first())

