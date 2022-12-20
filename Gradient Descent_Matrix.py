#LMS algorithm - Gradient Descent using Matrix derivatives
import csv
import random, math, numpy as np
from matplotlib import pyplot as plt
data_set = open("basketball.csv")
data = csv.reader(data_set)
x=[]
y=[]
next(data)
for k in data:
    x.append([1,float(k[0]),float(k[1]),float(k[2]),float(k[3]),float(k[4])])
    y.append([float(k[5])])
data_set.close()
x = np.array(x)
y = np.array(y)
def gd(x,y):
    b = np.matmul(x.transpose(), x)
    b = np.linalg.inv(b)
    c = np.matmul(b, x.transpose())
    theta = np.matmul(c, y)
    return theta
theta = gd(x,y)
print (theta)