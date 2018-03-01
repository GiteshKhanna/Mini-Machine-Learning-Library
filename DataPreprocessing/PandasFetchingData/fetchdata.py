import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[: , :-1].values
Y= dataset.iloc[: , -1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer.fit(X)
X= imputer.transform(X)

#Splitting the dataset into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
'''
#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)

#Plotting
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Exp')
plt.ylabel('Label')
plt.show()
