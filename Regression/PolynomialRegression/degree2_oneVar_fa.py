import numpy as np


def PolynomialFeatures(X):
    X = np.array(X)
    
    
    X = np.insert(X,0,np.reshape(X[:,0]*X[:,0],(1,len(X[:,0]))),axis=1)
    

    return X


'''
Z =([1],[2])

print(PolynomialFeatures(Z))
'''
