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
    theta = np.array([[1],[1],[1],[1],[1]])
    for epoch in range (1,100000):
        h=0
        err = y - np.dot(x,theta)
        temp = np.array([(np.sum(np.multiply(err,x),0))])
        alpha = 0.828134/1000000*math.exp(-epoch/25)
        theta = theta + alpha*(temp.transpose())
        cost = np.sum((err)**2)  
        if cost < min:
            thet = theta
    return (thet)
theta = gd(x,y)
print (theta)
    




    
