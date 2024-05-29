import math

def f(x):
    a = math.pow(x%6,2)
    b = a%7
    
    c=math.sin(x)
    result = b -c
    return result

for i in range(1,11):
    print(i, f(i))
