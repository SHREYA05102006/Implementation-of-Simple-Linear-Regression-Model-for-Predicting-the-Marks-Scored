# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
   
2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE,MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: V.Shreya
RegisterNumber:  212224230266
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\Downloads\\student_scores.csv")
df.head()
df.tail()
#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![1](https://github.com/user-attachments/assets/4c65f391-5a05-4488-af0e-ad6422cd65a7)
![2](https://github.com/user-attachments/assets/12b8f4cf-64f3-42dd-98fa-a52e6785c08e)
![3](https://github.com/user-attachments/assets/402a2a76-e4da-4ef6-8e1a-3a41fbd6f1cb)
![4](https://github.com/user-attachments/assets/8f70c4e9-1b41-4974-a90f-28b8fcc35242)
![5](https://github.com/user-attachments/assets/4bae9386-70fd-4747-8065-cfec01d514bd)
![6](https://github.com/user-attachments/assets/b61c9070-9823-4200-bf68-810226b21410)
![7](https://github.com/user-attachments/assets/1e9b0778-861b-4c60-b649-2d5574b1befc)
![8](https://github.com/user-attachments/assets/f09a6fe1-b2aa-4874-b89d-7b782d400fa0)
![9](https://github.com/user-attachments/assets/92d125e7-d7db-4869-a66c-eee36fcdca2d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
