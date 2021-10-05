# pyspark-assignment2

( I can't run the same scripts to get the results of task2 in big data set. Therefore, I could only provide the results of task1(in big data set), task2(in small data set), task3(in big data set).


## Describe here your project


In this assignment I will implement a k-nearest neighbors classifier (kNN classifier) in multiple
steps. KNN is an algorithm that can find the top-k nearest objects to a specific query object.
In this assignment we want to use kNN to classify text documents. Given a specific search text
like ”How many goals Vancouver score last year?” the algorithm can search in the document corpus
and find the top K similar documents.
We will create out of a text corpus a TF-IDF (Term Frequency - Inverse Document Frequency)
Matrix for the top 20k words of the corpus. The TF-IDF matrix will be used to compute similarity
distances between the given query text and each of the documents in the corpus.


# Submit your python scripts .py 

There are 3 final scripts.

# Other Documents. 

Screenshots for result are in docs folder.



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



