
## Project Description
 

4.1 Task 1 : Simple Linear Regression

4.2 Task 2 - Find the Parameters using Gradient Descent

4.3 Task 3 - Fit Multiple Linear Regression using Gradient Descent
use the following 5 features for this linear regression model are:
• Total working time in hours that a driver worked per day (a float number)
• Total travel distance in miles that a driver drove a taxi per day (a float number)
• Total number of rides per day (an integer number)
• Total amount of toll per day (a float number - toll amount indicates the number of rides over
the NYC bridges or rides to the airport.
• Total number of night rides between 1:00 AM and 6:00 AM per day

Bold driver method implementation.

## About the data

- A data set consisting of New York City Taxi trip reports in the Year 2013
https://chriswhong.com/open-data/foil_nyc_taxi/
- In this project, I would use a small dataset to show easy operations of RDD.
- Dataset address: gs://metcs777/taxi-data-sorted-small.csv.bz2
- Another small testing dataset address link: https://metcs777.s3.amazonaws.com/taxi-data-sorted-small.csv.bz2


# Submit python scripts .py 

Note: In main_task3.py, Bold-driver code is commented in the last part.

# Other Documents. 

I submit a gcp_results.pdf to show the results I got.

# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1.py 

```



```python

spark-submit main_task2.py 

```



```python

spark-submit main_task3.py 

```



