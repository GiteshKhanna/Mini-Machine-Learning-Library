import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first_var = (x[0]+2*x[1]-7)**2
    sec_var = (2*x[0] + x[1] - 5)**2

    return first_var + sec_var
    
    
    
