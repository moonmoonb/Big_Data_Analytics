from __future__ import print_function
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Firstly, I use the following PySpark 
# Code to cleanup the data and get the required field.
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import sys
from operator import add
import numpy as np


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


# In[3]:


# Then I remove all of the rides that have total amount of
# larger than 600 USD and less than 1 USD.


text_filtered = textlinesCorrected.filter(lambda x : float(x[16]) > 1).filter(lambda x : float(x[16]) < 600)
# text_filtered.top(2)


# task 3
collect_data = text_filtered.map(lambda x: ((x[1],x[2].split()[0]),
                                               (float(x[4])/3600,
                                                float(x[5]),
                                                1,
                                                float(x[15]),
                                                np.where(x[2][11:]>='01:00:00'and x[2][11:]<="06:00:00",1,0).tolist(),
                                                float(x[16]))))
data_group = collect_data.reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3],x[4]+y[4],x[5]+y[5]))
myRDD = data_group.map(lambda x: (x[1][-1],np.array(x[1][:-1])))

num_iteration = 400
learningRate = 0.0000000012
precision = 0.01
beta = np.ones(5)/10
size = data_group.count()
old_cost = 0
for i in range(num_iteration):
    gradientCost = myRDD.map(lambda x: (x[1], (x[0] - np.dot(x[1], beta)))).map(lambda x: (x[0] * x[1], x[1] ** 2)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    cost = gradientCost[1]
    # Stop if the cost is not descreasing
    if (abs(cost - old_cost) <= precision):
        print("Stopped at iteration", i)
        break
    old_cost = cost
    gradient = (-1 / float(size)) * gradientCost[0]
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient



# bold driver
# num_iteration = 100
# learningRate = 0.0001
# precision = 0.01
# beta = np.array([0.1]*5)
# size = data_group.count()
# old_cost = 0
# for i in range(num_iteration):
#     gradientCost = myRDD.map(lambda x: (x[1], (x[0] - np.dot(x[1], beta))))         .map(lambda x: (x[0] * x[1], x[1] ** 2)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
#     cost = gradientCost[1]
#     # Stop if the cost is not descreasing
#     if (abs(cost - old_cost) <= precision):
#         print("Stopped at iteration", i)
#         break
#     old_cost = cost
#     if (cost - old_cost) > 0:
#         learningRate = learningRate * 1.05
#     if (cost - old_cost) < 0:
#         learningRate = learningRate * 0.5
#     gradient = (-1 / float(size)) * gradientCost[0]
#     print(i, "Beta", beta, " Cost", cost)
#     beta = beta - learningRate * gradient



sc.stop()
