from copy import deepcopy,copy

class abc:
    def __init__(self,value1,value2):
        self.value1=value1
        self.value2=value2

obj1 = abc(1,3)
obj2 = deepcopy(obj1)
obj3 = copy(obj1)

obj2.value1 = 2
print(obj2.value1,obj1.value1)
obj3.value2 = 5
print(obj1.value2,obj3.value2)
