# pyspark

The goal of these code is to implement a k-nearest neighbors classifier (kNN classifier) in multiple
steps. KNN is an algorithm that can find the top-k nearest objects to a specific query object.

## Describe here your project

- In this assignment we want to use kNN to classify text documents. Given a specific search text
like ”How many goals Vancouver score last year?” the algorithm can search in the document corpus
and find the top K similar documents.
- We will create out of a text corpus a TF-IDF (Term Frequency - Inverse Document Frequency)
Matrix for the top 20k words of the corpus. The TF-IDF matrix will be used to compute similarity
distances between the given query text and each of the documents in the corpus.

## How to get the dataset: 
- Small dataset(Only 1000 Wikipedia pages), run in your laptop
- Small Data Set - Wikipedia Pages
- https://s3.amazonaws.com/metcs777/WikipediaPagesOneDocPerLine1000LinesSmall.txt.bz2
- Small Data Set - Wikipedia Page Categories
- https://metcs777.s3.amazonaws.com/wiki-categorylinks-small.csv.bz2
- 
- large dataset, run in google cloud platform
- Categories of the whole wikipedia: (store in google storage)
- gs://metcs777/wiki-categorylinks.csv.bz2
- 1 million pages - 2.2 GB, S3 URL: (store in google storage)
- gs://metcs777/WikipediaPagesOneDocPerLine1m.txt

## main_task1: Generate the Top 20K dictionary and Create the TF-IDF Array 
- Get the top 20,000 words in a local array and sort them based on the frequency of words. The
top 20K most frequent words in the corpus is our dictionary and we want to go over each document
and check if its words appear in the Top 20K words.

## main_task2: Implement the getPrediction function
- Implement kNN with getPrediction function.

## main_task3: Using Dataframes
- Use Spark Dataframe to provide summery statistics (max, average, median, std) about the number of wikipedia categories that are used for wikipedia pages. 
- Use Spark Dataframe to find the top 10 mostly used wikipedia categories. 

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



