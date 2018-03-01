import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    s = 0
    for i in x:
        s += abs(i*m.sin(i) + 0.1*i)

    return s
    
    
    
