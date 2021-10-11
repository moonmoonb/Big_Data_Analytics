#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import re
import numpy as np
import pandas as pd
import time
import psutil

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from numpy import dot
from numpy.linalg import norm
from operator import add
from pyspark.ml.linalg import Vectors
from pandas import Series,DataFrame


sc = SparkContext()
spark = SparkSession(sc)


# In[8]:


def buildArray(listOfIndices):
    
    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal


# In[9]:


d_corpus = sc.textFile(sys.argv[1], 1)
d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = d_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(lambda x,y: x + y)

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
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join(allWordsWithDocID)

# allDictionaryWords.take(1)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
# TF



# task 2 learning the model


# In[4]:


# rdd_X = allDocsAsNumpyArrays.map(lambda x: np.array(x[1]))
# # rdd_X.take(2)                                 
# rdd_y = allDocsAsNumpyArrays.map(lambda x: np.where(x[0][:2] == 'AU', 1, 0).tolist())   
# # rdd_y.take(2)
# traindata = rdd_y.zip(rdd_X)
# testdata = rdd_y.zip(rdd_X)


# In[5]:


# traindata.cache()
# train_size = traindata.count()
# # train_size # 3442


# In[6]:


# parameter_size = len(traindata.take(1)[0][1]) + 1
# # parameter_size # 20001


# In[7]:


# def LogisticRegression(traindata=traindata,
#                        max_iteration = 100,
#                        learningRate = 0.01,
#                        regularization = 0.01,
#                        precision = 0.01,
#                        optimizer = 'SGD' 
#                       ):

#     # initialization
#     prev_cost = 0
#     L_cost = []
#     prev_validation = 0
#     train_size = traindata.count()

#     parameter_size = len(traindata.take(1)[0][1]) + 1
#     parameter_vector = np.zeros(parameter_size)
# #     parameter_vector = np.zeros(parameter_size) # initialize with zeros
#     momentum = np.zeros(parameter_size)
#     prev_mom = np.zeros(parameter_size)
#     second_mom = np.array(parameter_size)
#     gti = np.zeros(parameter_size)
#     epsilon = 10e-8
    
#     for i in range(max_iteration):

#         bc_weights = parameter_vector[:-1]
#         bc1_weights = parameter_vector[-1]


#         res = traindata.treeAggregate((np.zeros(parameter_size), 0, 0),\
#               lambda x, y:(x[0]\
#                           + (np.append(y[1], 1)) * (-y[0] + (np.exp(np.dot(y[1], bc_weights) + bc1_weights)\
#                           /(1 + np.exp(np.dot(y[1], bc_weights) + bc1_weights)))),\
#                           x[1] \
#                           + y[0] * (-(np.dot(y[1], bc_weights) + bc1_weights)) \
#                           + np.log(1 + np.exp(np.dot(y[1],bc_weights)+ bc1_weights)),\
#                           x[2] + 1),
#               lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2]))

#         cost =  res[1]

#         # calculate gradients
#         gradient_derivative = (1.0 / res[2]) * res[0]
        
#         if optimizer == 'SGD':
#             parameter_vector = parameter_vector - learningRate * gradient_derivative


#         print("Iteration No.", i, " Cost=", cost)
        
#         # Stop if the cost is not descreasing
#         if abs(cost - prev_cost) < precision:
#             print("cost - prev_cost: " + str(cost - prev_cost))
#             break
#         prev_cost = cost
#         L_cost.append(cost)
        
#     return parameter_vector, L_cost


# In[ ]:


# parameter_vector_sgd, L_cost_sgd = LogisticRegression(traindata=traindata,
#                        max_iteration = 100,
#                        learningRate = 0.8,
#                        precision = 0.01,
#                        optimizer = 'SGD' 
#                       )


# In[ ]:


# task 2


# In[10]:


myRDD = allDocsAsNumpyArrays.map(lambda x: (np.array(x[1]), np.where(x[0][:2] == 'AU', 1, 0)))
# myRDD.take(2)
#
#
# # In[30]:
#
#
# # Without Regularization
# learningRate = 0.1
# num_iteration = 100
#
# myRDD.cache()
#
# allDocsAsNumpyArrays.cache()
#
# precision = 0.01
#
# oldCost = 0
#
# size = allDocsAsNumpyArrays.count()
#
# beta = np.zeros(20000)
#
# cost_list = []
#
# k = 0
#
# for i in range(num_iteration):
#     # first map : theta, x, y, e^theta
#     gradientCost = myRDD.map(lambda x : ((np.dot(x[0], beta)), x[0], x[1], np.exp(np.dot(x[0], beta))))                    .map(lambda x : ((-x[1]) * (x[2] - (x[3]/(1 + x[3]))), (- x[2] * x[0] + np.log(1 + x[3]))))                    .reduce(lambda x, y : (x[0] + y[0], x[1] + y[1]))
#
#     cost = gradientCost[1]
#
#     gradient = gradientCost[0]
#
#     cost_list.append(cost)
#
#     print(i, " Cost", cost)
#     beta = beta - learningRate * gradient
#
#     # Stop if the cost is not descreasing
#     if(abs(cost - oldCost) <= precision):
#         print("Stoped at iteration", i)
#         break
#
#     oldCost = cost
#
#
#
# # In[ ]:
#

# Regularization test
learningRate = 0.01
num_iteration = 200

myRDD.cache()

allDocsAsNumpyArrays.cache()

precision = 0.5

oldCost = 0

size = allDocsAsNumpyArrays.count()

beta = np.zeros(20000)

cost_list_r = []

k = 0

for i in range(num_iteration):
    # first map : theta, x, y, e^theta
    gradientCost = myRDD.map(lambda x : ((np.dot(x[0], beta)), x[0], x[1], np.exp(np.dot(x[0], beta))))                    .map(lambda x : ((-x[1]) * (x[2] - (x[3]/(1 + x[3]))), (- x[2] * x[0] + np.log(1 + x[3]))))                    .reduce(lambda x, y : (x[0] + y[0], x[1] + y[1]))
    
    cost = gradientCost[1]
    
    # 0.1 as regulazition penalty
    gradient = gradientCost[0] + 0.1 * np.linalg.norm(beta)
    
    cost_list_r.append(cost)
        
    if i%10 == 0 :
            print("Iteration No.", i, " Cost=", cost)
            
    beta = beta - learningRate * gradient
    
    # Stop if the cost is not descreasing       
    if(abs(cost - oldCost) <= precision):
        print("Stoped at iteration", i)
        break
        
    oldCost = cost

# Print out the five words with the largest regression coefficients
d = beta.argsort()[-5:][::-1]
for i in range(0, 5):
    print(dictionary.filter(lambda x : (x[1] == d[i])).collect())
