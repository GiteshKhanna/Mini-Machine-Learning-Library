import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first_var =abs(x[0]**2 + x[1]**2 +x[0]*x[1])
    sec_var = abs(m.sin(x[0]))
    third_var = abs(m.cos(x[1]))

    return first_var+sec_var+third_var
    
    
    
