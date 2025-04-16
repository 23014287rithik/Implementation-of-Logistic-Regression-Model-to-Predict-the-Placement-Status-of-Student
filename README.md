# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RITHIK V
RegisterNumber: 212223230171
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

## top 5 elements
![image](https://github.com/user-attachments/assets/1cf48b72-0a30-47d5-b3de-4dfbc467e2b3)

![image](https://github.com/user-attachments/assets/7eb2c8d7-39ba-4ca4-9106-9399e4e8ff43)

![image](https://github.com/user-attachments/assets/6a852ca5-6eea-48c0-9742-f3064206067f)

## DATA DUPLICATE
![image](https://github.com/user-attachments/assets/00694b19-d000-406e-87e5-181e597dc280)

## print data

![image](https://github.com/user-attachments/assets/f90be970-1701-4796-af71-f340ea15738a)

## data_status
![image](https://github.com/user-attachments/assets/e84dca65-2609-4a73-ad30-c523ad3549e2)

## y pridiction array
![image](https://github.com/user-attachments/assets/786d6bbe-8365-4eeb-be39-3ed659f80f0c)

## confusion array

![image](https://github.com/user-attachments/assets/cd4b36d6-4db8-454f-9126-362d6bb01d91)

## accurate value

![image](https://github.com/user-attachments/assets/8bf7403a-e6b2-4038-9274-0f6be4dfde21)

## classification report

![image](https://github.com/user-attachments/assets/2e2da379-b50d-42b3-bd5f-84cffb8fde3e)

## pridiction

![image](https://github.com/user-attachments/assets/4fd7e203-ec9d-43df-addf-3eb4d2b66728)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
