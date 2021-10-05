# pyspark

The goal of these code is to implement a set of Spark program in python(using Apache Spark).

## About the data

- A data set consisting of New York City Taxi trip reports in the Year 2013
https://chriswhong.com/open-data/foil_nyc_taxi/
- In this project, I would use a small dataset to show easy operations of RDD.
- Dataset is attached in the folder: taxi-data-sorted-small.csv.bz2
- Also, I attach a attributes information: Taxi_Data_Set_Field.csv

## Describe here your project

1. Firstly, I would make data processing. 
2. The first job is to find the top-10 active taxis
3. Finally, I would like to figure out who the top 10 best drivers are in terms of their average earned money per minute spent carrying a customer.
 


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



