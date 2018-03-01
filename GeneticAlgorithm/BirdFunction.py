import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first_var = m.sin(x[0])*m.exp((1-m.cos(x[1]) )**2)
    sec_var = m.cos(x[1])*m.exp((1-m.sin(x[0]))**2)
    third_var = (x[0] - x[1])**2

    return first_var + sec_var + third_var
    
    
    
