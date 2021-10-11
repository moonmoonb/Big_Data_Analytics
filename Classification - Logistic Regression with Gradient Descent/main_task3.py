#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import re
import math 
import sys
import numpy as np
from operator import add
from pyspark import SparkConf
from pyspark import SparkContext
import findspark
findspark.init()

conf=SparkConf()
sc = SparkContext()


# In[ ]:


def buildArray(listOfIndices):
    
    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal


# In[ ]:


d_corpus = sc.textFile(sys.argv[1], 1)
d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
allWords = d_keyAndListOfWords.flatMap(lambda x:x[1]).map(lambda x: (x, 1))
allCounts = allWords.reduceByKey(add)
topWords = allCounts.top(20000,key=lambda x: x[1])
topWordsK = sc.parallelize(range(20000))
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allDictionaryWords = dictionary.join(allWordsWithDocID)
justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
allDocsAsNumpyArrays=allDictionaryWordsInEachDoc.map(lambda x: (x[0],  buildArray(x[1])))


# In[ ]:


myRDD = allDocsAsNumpyArrays.map(lambda x: (np.where(x[0][:2] == 'AU', 1, 0), np.array(x[1])))


# In[ ]:


parameter_vector = np.ones(20000)/100
maxIteration=200
iter=0
myRDD.cache()
precision=0.00001
regularization=0.01   
learning_rate=0.0001
prev_cost=0
while iter<maxIteration:
    res = myRDD.treeAggregate((np.zeros(20000), 0, 0),                  lambda x, y:(x[0]                              + (y[1]) * (-y[0] + (np.exp(np.dot(y[1], parameter_vector))                              /(1 + np.exp(np.dot(y[1], parameter_vector) )))),                              x[1]                               + y[0] * (-(np.dot(y[1], parameter_vector)))                               + np.log(1 + np.exp(np.dot(y[1],parameter_vector))),                              x[2] + 1),
                  lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2]))
    #without regularization
    cost =  res[1] + regularization * (np.square(parameter_vector).sum())
    LLH=-res[1]
    gradient_derivative = (1.0 / res[2]) * res[0] + 2 * regularization * parameter_vector
    prev_vector = parameter_vector
    prev_cost=cost
    parameter_vector = parameter_vector - learning_rate * gradient_derivative
    if cost>prev_cost:
            learning_rate=0.5*learning_rate
    else:
            learning_rate=1.05*learning_rate
    if np.sum((parameter_vector - prev_vector)**2)**0.5 < precision:
            print("Stoped at iteration", iter)
            break
    if iter%2==0:
        print("Iteration No.", iter, " Cost=", cost)
    iter+=1


# In[ ]:


#task3
test=sc.textFile(sys.argv[2], 1)
d_test = test.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')
d_testOfWords = d_test.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))        
testWords = d_testOfWords.flatMap(lambda x:x[1]).map(lambda x: (x, 1))
testCounts = testWords.reduceByKey(add)
testWordsWithDocID = d_testOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
testDictionaryWords = dictionary.join(testWordsWithDocID)
testDocAndPos =testDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
testDictionaryWordsInEachDoc = testDocAndPos.groupByKey()

testDocsAsNumpyArrays=testDictionaryWordsInEachDoc.map(lambda x: (x[0],  buildArray(x[1])))
test_data=testDocsAsNumpyArrays.map(lambda x:(x[0],(np.where(x[0][:2]=='AU',1,0).tolist(),x[1])))
prediction=test_data.map(lambda x: (x[0],np.where(np.dot(x[1][1],parameter_vector)>0,1,0).tolist()))
p_a_pair=prediction.join(test_data.map(lambda x:(x[0],x[1][0])))
TP=p_a_pair.filter(lambda x: x[1]==(1,1)).count()
FP=p_a_pair.filter(lambda x:x[1]==(1,0)).count()
FN=p_a_pair.filter(lambda x:x[1]==(0,1)).count()

print('The true positive in confusion matrix:', TP)
print('The false negative in confusion matrix:', FN)
print('The false positive in confusion matrix:', FP)

recall=TP/(TP+FN)
precision=TP/(TP+FP)
print("F1 score:",2*(precision*recall)/(precision+recall))
sample_FP=p_a_pair.filter(lambda x:x[1]==(1,0)).map(lambda x: x[0]).top(3)
print('The 3 False Positive samples ID are:', sample_FP)
content=d_test.filter(lambda x: x[0] in sample_FP).map(lambda x:x[1]).collect()
for i in content:
    print(content)


