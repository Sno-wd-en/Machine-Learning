#Logistic Regression on Loan_Data.csv. Accuracy = 70 %
import csv
import math, numpy as np
data_set = open("Loan_Data.csv")
data = csv.reader(data_set)
x = []
y = []
next(data)
def gender(a):
    if a=="Male":
        return 1
    else:
        return 0
def married(a):
    if a=="Yes":
        return 0
    return 1
def dependent(a):
    if a=="":
        return 0
    if a=="3+":
        return 4
    else:
        return int(a)
def education(a):
    if a[0]=="G":
        return 1
    else:
        return 0
def selfemp(a):
    if a=="Yes":
        return 1
    else:
        return 0
def appinc(a):
    return int(a)
def coapp(a):
    if a:
        return float(a)
    else:
        return 0
def loanamt(a):
    if a:
        return int(a)
    else:
        return 0
def loant(a):

    if a:
        return int(a)
    else:
        return 0
def credithist(a):
    if a:
        return int(a)
    else:
        return 0
def property(a):
    if a[0]=="U":
        return 1
    if a[0]=="S":
        return 2
    if a[0]=="R":
        return 3
def status(a):
    if a=="Y":
        return 1
    else:
        return 0
for k in data:
    x.append([1,gender(k[1]), married(k[2]), dependent(k[3]), education(k[4]), selfemp(k[5]), 
    appinc(k[6]), coapp(k[7]), loanamt(k[8]), loant(k[9]), credithist(k[10]), property(k[11])])
    y.append([status(k[12])])
data_set.close()
x = np.array(x)
y = np.array(y)
def g(x:int)->float:
    return (1/(1+math.exp(-x)))
def logistic_regression(x,y):
    mintheta = theta = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
    min = float('inf')
    mink = 1
    for k in range (1,100000):
        v = np.dot(x,theta)
        v = 1/(1+np.exp(-v))
        v[v>=0.5]=1
        v[v<0.5]=0
        err = y - v
        alpha = 96/10000
        cost = np.sum(err**2)
        temp = np.array([(np.sum(np.multiply(err,x),0))])
        theta = theta + alpha*(temp.transpose())
        if cost < min:
            mintheta = theta
            min = cost
            mink = k
    return (mintheta)
vector = logistic_regression(x,y)
print (vector)