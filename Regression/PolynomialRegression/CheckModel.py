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
        '''
        print("Hypothesis")
        print(self.trainingSet.dot(self.theta))
        print('++++++++++++++++++')
        '''
        return (self.trainingSet.dot(self.theta))


##constant(x0*theta0) + x1^2(theta1) + x2^2(theta2) + ....
    def costFunction(self):
        h = self.hypothesis()
        sub = np.subtract(h,self.dependentSolSet)
        return (sum(sub**2))/(2*len(sub))

    def makeModel(self,learning_rate=0.02):
        y = self.dependentSolSet
        x = self.trainingSet
        m = len(y)
        iter = 0
        

        while(iter!=1000):
            
            delta=np.array(self.hypothesis()-y)
            #Updating Theta[0]--(Because Derivative is different)
            self.theta[0]= self.theta[0] - (learning_rate*(sum(delta)))
            #Updating all others except Theta[0]
            for i in range(1,len(self.theta)):
                self.theta[i]= self.theta[i] - (learning_rate*(sum(delta*(x[:,i]))))/m

            iter+=1
            '''
        print('Theta'
        print(self.theta)
        print('_____________________')
        '''
            
        
    def predict(self,testSet):
        testSet=np.c_[np.ones(len(testSet)),np.array(testSet)]
        return (testSet.dot(self.theta))


##################################################################################
#TESTING
##################################################################################


#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[: , 1:2].values
Y= dataset.iloc[: , -1].values
'''
#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer.fit(X)
X= imputer.transform(X)
'''
'''
#Splitting the dataset into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 0)

'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X)

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

##ML
import degree2_oneVar_fa as ff
X_mypoly=ff.PolynomialFeatures(X_train)
Object = Linear_Regression(X_mypoly,Y)
Object.makeModel(learning_rate=0.01)

print('Real Solution:')
print(Y)
print('Predicted Solution')
print(Object.predict(X_mypoly))


# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
#plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.plot(X,Object.predict(X_mypoly),color ='green')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


'''
# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
#plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
#plt.plot(X,Object.predict(X),color = 'pink')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''






