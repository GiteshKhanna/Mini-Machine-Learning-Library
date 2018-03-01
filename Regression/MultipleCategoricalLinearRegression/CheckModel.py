import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Linear_Regression:
    def __init__(self,X,Y):
        self.trainingSet=np.c_[np.ones(len(X)),X]
        self.dependentSolSet=np.array(Y)
        self.no_of_attributes = 0
        
        for i in range(len(X)):
            self.no_of_attributes = max(self.no_of_attributes,len(self.trainingSet[i]))
            
        self.theta = np.array([1.00 for i in range(self.no_of_attributes)])

    def hypothesis(self):
        return (self.trainingSet.dot(self.theta))


##x1*theta1 + x2*theta2 + x3*theta3 + ....
    def costFunction(self):
        h = self.hypothesis()
        sub = np.subtract(h,self.dependentSolSet)
        return (sum(sub**2))/(2*len(sub))

    def makeModel(self,learning_rate=0.2):
        y = self.dependentSolSet
        x = self.trainingSet
        m = len(y)
        iter = 0

        while(iter!=5000):
            
            delta=np.array(self.hypothesis()-y)
            #Updating Theta[0]--(Because Derivative is different)
            self.theta[0]= self.theta[0] - (learning_rate*(sum(delta)))
            #Updating all others except Theta[0]
            for i in range(1,len(self.theta)):
                self.theta[i]= self.theta[i] - (learning_rate*(sum(delta*x[:,i])))/m

            iter+=1
            #print(self.theta)
            #print('_____________________')
            
        
    def predict(self,testSet):
        testSet=np.c_[np.ones(len(testSet)),np.array(testSet)]
        return (testSet.dot(self.theta))



#Importing dataset
dataset = pd.read_csv('Dataset-mtcars.csv')
X= dataset.iloc[: , 2:5].values
Y= dataset.iloc[: , 1].values

'''
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding dummy variable Trap
X=X[:,1:]
'''
#Splitting the dataset into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

print(X_train)

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
print('Predefined Regressor Solution:')
print(Y_pred)


##ML
Object = Linear_Regression(X_train,Y_train)
Object.makeModel(learning_rate=0.0001)
print('Real Solution:')
print(Y_test)
print('Predicted Solution')
print(Object.predict(X_test))

'''
#Plotting
plt.scatter(X_train[:,0],Y_train,color='red')
plt.scatter(X_test[:,0],Y_test,color='green')
plt.plot(X_train[:,0],Object.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Exp')
plt.ylabel('Label')
plt.show()
'''







