'''
Anomaly Detection:
When an input varies from normal behaviour.
Eg. Flaw detection in airplane engines.
Supervised Learning can also be used. But anomaly detection
is used when our dataset contains less number of examples having
anomaly set to 1.
If there are comparable number of 1/0 , then supervised learning
will be a better choice

Here, 1 implies Anomaly detected
'''
import numpy as np
import math as m
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

class AnomalyDetection:

    def __init__(self,X):
        self.Set= np.array(X)
        self.mean= sum(self.Set)/len(self.Set)
        self.variance= sum((self.Set-self.mean)**2)/len(X)

#Returns probability of a Set of examples to be normal
#Accepts a 2D array only
    def probability(self,X):
        print('Mean:',self.mean)
        print('variance:',self.variance)
        print('right',(np.exp(-(X-self.mean)**2/(2*self.variance))))
        prob = (np.exp(-(X-self.mean)**2/(2*self.variance)))/(np.sqrt(2*m.pi*self.variance))
        print("Probability",prob,"\n")
        prod = np.prod(prob,axis=1)
        return prod

    def vis_prob(self,X):
        return (np.exp(-(X-self.mean)**2/(2*self.variance)))/(np.sqrt(2*m.pi*self.variance))

#Returns a prediction list of the examples sent in 2D array
    def predict(self,X,epsilon):
        probability = self.probability(X)
        print(probability)
        pred_list = []
        
        for i in range(0,len(X)):
            if(probability[i]<epsilon):
                pred_list.append(1)
            else:
                pred_list.append(0)

        return pred_list
            
            
        
#Creates a histogram of frequencies of occurences in each dimension
# 5 dimension set = 5 histograms indicating frequencies in each.
    def visualizeHist(self,num_bins=2):
        for i in range(0,len(self.Set)-1):
            plt.hist(self.Set[:,i], bins=num_bins, facecolor='blue', alpha=0.5)
            plt.show()
'''
    def visualizeCurve(self,num_bins=2):
        
        
        x = np.linspace(self.mean - 3*np.sqrt(self.variance), np.mean + 3*np.sqrt(self.variance), 100)
        plt.plot(x,mlab.normpdf(self.Set, mu, sigma))
        plt.show()
            
'''
    

    


######################################################################
#TESTING
######################################################################
#Importing dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data.csv')
X= dataset.iloc[: ,1 : 11].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:11])
X[:, 1:11] = imputer.transform(X[:, 1:11])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 99)


A = AnomalyDetection(np.log(X_train))
A.visualizeHist(20)
print(A.predict(X_test,0.002))


'''
X = [[8,2,3],[2,3,4],[6,8,10],[6,3,6],[8,2,3],[4,2,3],[1,2,3],[1,2,3]]        
A = AnomalyDetection(X)
print(A.predict([[1,2,3],[4,5,6]],0.002))
A.visualizeCurve()
'''

    
