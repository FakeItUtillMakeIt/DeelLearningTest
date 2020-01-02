import numpy as np

x=np.array([4,8,9,8,7,12,6,10,6,9])
y=np.array([9,20,22,15,17,23,18,25,10,20])
##x=np.ndarray(x)
##y=np.ndarray(y)

k=((x.mean()*y.mean())-(x*y).mean())/((x.mean())**2-(x**2).mean())
b=y.mean()-k*x.mean()

print(k,b)
print(2*k+b)
