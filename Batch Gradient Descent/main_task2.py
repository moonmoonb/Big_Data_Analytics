from __future__ import print_function

# Firstly, I use the following PySpark
# Code to cleanup the data and get the required field.

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import sys

sc = SparkContext()
spark = SparkSession(sc)

lines = sc.textFile(sys.argv[1])
taxilines = lines.map(lambda x: x.split(','))

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    
def correctRows(p):
    if(len(p) == 17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[5]) != 0 and isfloat(p[11]) != 0):
                return p
            
textlinesCorrected = taxilines.filter(correctRows)


# In[9]:


# Then I remove all of the rides that have total amount of
# larger than 600 USD and less than 1 USD.


# In[25]:


text_filtered = textlinesCorrected                .filter(lambda x : float(x[16]) > 1)                .filter(lambda x : float(x[16]) < 600)
# text_filtered.top(2)
size = text_filtered.count()


# In[14]:


textfilered = text_filtered.map(lambda x: (x[5], x[11]))


# In[17]:


myRDD = textfilered.map(lambda x: (float(x[1]), np.array(float(x[0]))))
# myRDD.take(2)


# In[54]:


# Now we do gradient Decent on our RDD data set. 
from pyspark.ml.linalg import Vectors
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import time
import psutil

learningRate = 0.000000012
num_iteration = 100 
max_iterarion = 400


beta = np.ones(1)/10

textfilered.cache()

precision = 0.01

oldCost = 0

cost_list = []

k = 0

for i in range(num_iteration):
    
    gradientCost=myRDD.map(lambda x: (x[1], (x[0] - x[1] * beta) ))                        .map(lambda x: (x[0]*x[1], x[1]**2 ))                        .reduce(lambda x, y: (x[0] +y[0], x[1]+y[1] ))
    
    cost= gradientCost[1]
    
    
    
    gradient=(-1/size)* gradientCost[0]
    
    if(i > max_iterarion):
        break
        
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient
    
    # Stop if the cost is not descreasing 
    if(abs(cost - oldCost) <= precision):
        print("Stoped at iteration", i)
        break
        
    
    oldCost = cost

sc.stop()
