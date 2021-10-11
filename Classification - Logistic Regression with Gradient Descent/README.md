
## Describe here your project


4.1 Task 1 : Data Preparation(TF)

show the frequency position of the words “applicant”, “and”, “attack”, “protein”, and “car”

4.2 Task 2 - Learning the Model

the five words with the largest regression coefficients.

4.3 Task 3 - Task 3 - Evaluation of the learned Model 

use the model to predict whether or not each of the testing points correspond to Australian court cases. 

compute the f1-score by classifier


# Submit your python scripts .py 

Note: 
1. In main_task2.py, gradient descent algorithm is implemented by adding regularization penalty. Gradient descent algorithm without regularization is commented.
2. In main_task3.py, I use gradient descent algorithm without regularization.

# Other Documents. 

I submit a assignment4_result.pdf to show the results I got.

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



