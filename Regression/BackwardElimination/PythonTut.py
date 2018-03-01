import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X= dataset.iloc[: , :-1].values
Y= dataset.iloc[: , -1].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable Trap
X=X[:,1:]


#Splitting the dataset into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)


#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)



#Backward Elimination
import statsmodels.formula.api as sm
# Adding ones to X
X = np.c_[np.ones(len(X)),X]
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt=X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
print(regressor_OLS.summary())

