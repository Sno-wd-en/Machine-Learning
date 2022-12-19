#LMS algorithm - Gradient Descent
import csv
import random, math, numpy as np
from matplotlib import pyplot as plt
data_set = open("basketball.csv")
data = csv.reader(data_set)
x=[]
y=[]
next(data)
for k in data:
    x.append([1,float(k[0]),float(k[1]),float(k[2]),float(k[3])])
    y.append([float(k[4])])
data_set.close()
x = np.array(x)
y = np.array(y)
def gd(x,y):
    min = float('inf')
    thet = 0
    minepoch = 1
    theta = np.array([[1,1,1,1,1]])
    for epoch in range (1,10000):
        h=0
        err = y - np.matmul(x,np.transpose(theta))
        for k in range (len(x)):
            h += err[k][0]*x[k]
        cost = np.sum((err)**2)
        alpha = 0.828134/1000000*math.exp(-epoch/25)
        theta = theta + alpha*h
        if cost < min:
            thet = theta
            min = cost
            minepoch = epoch
    return (min,thet,minepoch)
theta = gd(x,y)
print (theta)
    




    
