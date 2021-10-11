from __future__ import print_function
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



# Then I remove all of the rides that have total amount of
# larger than 600 USD and less than 1 USD.




text_filtered = textlinesCorrected                .filter(lambda x : float(x[16]) > 1)                .filter(lambda x : float(x[16]) < 600)
# text_filtered.top(2)



# Task 1: Simple Linear Regression
textfilered = text_filtered.map(lambda x: (x[5], x[11]))
                
# textfilered.top(2)

n = textfilered.count() # total data
xiyi = textfilered.map(lambda x: (float(x[0])*float(x[1]))).reduce(lambda x, y: x+y)
sum_xi = textfilered.map(lambda x: float(x[0])).reduce(lambda x, y: x+y)
sum_yi = textfilered.map(lambda x: float(x[1])).reduce(lambda x, y: x+y)
sum_xi_square = textfilered.map(lambda x: float(x[0])**2).reduce(lambda x, y: x+y)
sum_yi_square = textfilered.map(lambda x: float(x[1])**2).reduce(lambda x, y: x+y)


# In[40]:


m_hat = (n * xiyi - sum_xi * sum_yi) / (n * sum_xi_square - sum_xi**2)
b_hat = ((sum_xi_square * sum_yi) - sum_xi * xiyi) / (n * sum_xi_square - sum_xi**2)
print(m_hat)
print(b_hat)

sc.stop()


