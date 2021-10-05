from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName='Taxi-Assignment')
    lines = sc.textFile(sys.argv[1], 1)
    taxilines = lines.map(lambda x: x.split(','))


    def isfloat(value):


        try:
            float(value)
            return True
        except:
            return False


    def correctRows(p):
        if (len(p) == 17):
            if isfloat(p[5]) and isfloat(p[11]):
                if (float(p[5]) != 0 and isfloat(p[11]) != 0):
                    return p


    textlinesCorrected = taxilines.filter(correctRows)

    # 4.1
    taxi_driver_distinct = textlinesCorrected.map(lambda x: (x[0], x[1])).distinct()
    taxi_driver = taxi_driver_distinct.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y)

    taxi_driver.saveAsTextFile(sys.argv[2])
    output = taxi_driver.top(10, lambda x: x[1])

    sc.stop()
