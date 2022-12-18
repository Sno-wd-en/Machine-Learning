#LMS algorithm - Gradient Descent
import csv
import random, math, numpy as np
from matplotlib import pyplot as plt
data_set = open("Salary_Data.csv")
data = csv.reader(data_set)
x = []
y=[]
next(data)
for k in data:
    x.append(float(k[0]))
    y.append(float(k[1]))
data_set.close()
def gd(x,y):
    min = float('inf')
    thet = 0
    for epoch in range (1,10):
        while True:
            r = int(random.random()*(len(x)-1))
            s = int(random.random()*(len(x)-1))
            try:
                theta = np.array([(y[r]*x[s]-x[r]*y[s])/(x[s]-x[r]),(y[r]-y[s])/(x[r]-x[s])])
                break
            except ZeroDivisionError:
                pass
        for t in range(1,100):
            err = 0
            e = 0
            alpha = 0.004*math.exp(-t)
            for k in range (len(x)):
                h = theta[0]+x[k]*theta[1]
                h = theta[0]+x[k]*theta[1]

                err += ((y[k] - h)*x[k])
                e += 0.5*((h-y[k])**2)
            theta = theta + alpha*err
            if e < min:
                thet = theta
                min = e
    return (min,thet)
theta = gd(x,y)
print (theta)
y1 = []
for i in x:
    y1.append(theta[1][0]+theta[1][1]*i)
plt.plot(x,y)
plt.plot(x, y1)
# plt.plot(x,y2)
plt.show()




    
