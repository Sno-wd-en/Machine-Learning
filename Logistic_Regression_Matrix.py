import csv
import time, math, numpy as np
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
        return 1
    return 0
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
    if a[0]=="Y":
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
def logistic_matrix(x,y):
    b = np.matmul(x.transpose(), x)
    b = np.linalg.inv(b)
    c = np.matmul(b, x.transpose())
    theta = np.matmul(c, y)
    err = np.sum((y - np.dot(x,theta))**2)
    return (theta,err)
theta = logistic_matrix(x,y)
print (theta)