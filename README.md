# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KISHORE B
RegisterNumber: 212223240073
*/
```
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:
### Data Head:
![318642657-92e9d8f7-78b7-41f4-b441-edb4279d8f0d](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/5eef2abc-af52-4680-a9a6-981c17f9c0d6)


### Data Info:
![318642666-611c2f84-09e5-4040-9ec8-f7e2c08175a8](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/f5902a3b-2954-46d4-b6b2-d527ec4731d6)



### isnull() sum():
![318642685-1f0c3c0c-5949-4a80-a74b-639171f3fa5a](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/3c019e05-e7af-4c80-ab2a-1bfe51037e52)


### Data Head for salary:
![318642693-dcba8525-df04-48ef-a97d-faf2b103c6a1](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/a0174370-74ff-43d8-a253-41e54996899a)


### Mean Squared Error:
![318642723-f51e8787-8412-41da-b05d-b94a52ab5437](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/bc3f3a7a-8822-401b-af9c-dc20a8bf85d2)

  

### r2 Value:
![318642796-c5e08f65-efd0-487f-8b20-e236dc993fd2](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/4b33127f-1422-4c56-b277-701b853d219e)


### Data Prediction:
![318642857-3181b25d-30f0-4d1b-8087-0ec315cec41c](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139122/8005a6b3-9da9-4199-a0e4-11a389f5b470)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
