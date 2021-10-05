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

    # 4.2
    # First, I create a new rdd with total duration of the trip(in minutes) for each driver
    driver_total_duration_min = textlinesCorrected.map(lambda x: (x[1], int(x[4]) / 60)).reduceByKey(lambda x, y: x + y)

    # Then, I create a new rdd with total amount of each driver got paid
    driver_total_paid = textlinesCorrected.map(lambda x: (x[1], float(x[16]))).reduceByKey(lambda x, y: x + y)

    # The last step is to join these 2 rdd. Then calculate the best average 
    best_average_earn = driver_total_duration_min.join(driver_total_paid).filter(lambda x: x[1][0] != 0).mapValues(
        lambda x: x[1] / x[0])

    best_average_earn.saveAsTextFile(sys.argv[2])

    output = best_average_earn.top(10, lambda x: x[1])

    sc.stop()
