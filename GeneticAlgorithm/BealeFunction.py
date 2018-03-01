import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first_var = (1.5-x[0]+x[0]*x[1])**2
    sec_var = (2.25-x[0] + x[0]*(x[1]**2))**2
    third_var = (2.625-x[0]+x[0]*((x[1])**3))**2

    return first_var + sec_var + third_var
    
    
    
