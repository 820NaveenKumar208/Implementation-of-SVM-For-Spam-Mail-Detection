
# Develop By : NaveenKumar.T
# Reg No : 212223220067
# Implementation-of-SVM-For-Spam-Mail-Detection

# AIM:
To write a program to implement the SVM For Spam Mail Detection.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1.Import the necessary packages using import statement.
2.Read the given csv file and print the number of contents to be displayed. 
3.Split the dataset using train_test_split. 
4.Calculate Y_Pred and accuracy. 
5.Display the result.
## Program:
```

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  

```

# Output:
## Data.head():

![174660497-82d7f30e-c8da-423f-b298-6e3208a42595](https://github.com/820NaveenKumar208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/154746066/e844d416-4c80-4e41-a268-48daa412504c)

## Data.info():
![174660571-71df2803-7e89-41d8-93b8-b6862a12eee4](https://github.com/820NaveenKumar208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/154746066/05094a54-ec40-4f8e-9fb6-abda32c974cd)

## Accuracy:
![174660764-eecc1fce-2712-48c1-a8c8-7a83280e987c](https://github.com/820NaveenKumar208/Implementation-of-SVM-For-Spam-Mail-Detection/assets/154746066/975b756c-807d-455f-8a90-872a1b7b2ff8)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
