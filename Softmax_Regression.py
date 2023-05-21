import numpy as np
import tensorflow as tf
import time
import math, csv, pandas as pd
 
def prob(theta, x, y):
    num1 = math.exp(np.dot(theta[0],x))
    num2 = math.exp(np.dot(theta[1],x))
    if y==1:
        return (num1/(1+num1+num2))
    elif y==2:
        return (num2/(1+num1+num2))
    else:
        return (1/(1+num1+num2))

def species(s):
    if s=="Iris-setosa":
        return 1
    elif s=="Iris-versicolor":
        return 2
    elif s=="Iris-virginica":
        return 3

df = pd.read_csv('Iris.csv')

df1=df.loc[:, df.columns.drop(['Id', 'Species'])]
df2 = pd.DataFrame().assign(Id=df['Id'], Species=df['Species'])
df1["const"] = 1
#print(df1.to_string())
#df1.rename(columns = {'SepalLengthCm':1, 'SepalWidthCm':2,'PetalLengthCm':3, 'PetalWidthCm':4}, inplace = True)
conditions = [
    (df['Species'] == "Iris-setosa"),
    (df['Species'] == "Iris-versicolor"),
    (df['Species'] == "Iris-virginica")
    ]

values = [1,2,3]
df['tier'] = np.select(conditions, values)

theta1 = np.array([[1],[1],[1],[1],[-10]])
theta2 = np.array([[1],[1],[1],[1],[-10]])
# print(theta1)

data_set = open("Iris.csv")
data = csv.reader(data_set)
x=[]
y=[]
next(data)
for k in data:
    x.append([1,float(k[1]),float(k[2]),float(k[3]),float(k[4])])
data_set.close()
x = np.array(x)
y = df['tier'].to_numpy()

totaltime = time.time()
maxc = 0
maxj = 0
maxl = 0
for j in range(1,100):
    elapse = time.time()
    theta1 = np.array([[1],[1],[1],[1],[-10]])
    theta2 = np.array([[1],[1],[1],[1],[-10]])

    # if j!=13:
    #     continue

    for epoch in range(1000):
        dot1 = np.dot(x,theta1)
        dot2 = np.dot(x,theta2)
        # print ("Epoch : %2d" % (epoch+1))
        sum1 = np.array([[0,0,0,0,0]])
        sum2 = np.array([[0,0,0,0,0]])
        l=1
        a = j*0.01*math.exp(-epoch/1000)
        for i in range (len(x)):
            # t1 = math.exp(dot1[i]-dot2[i])
            if (dot1[i]-dot2[i])>20:
                c2 = 0
                if (dot1[i]<-20):
                    c1 = 0
                else:
                    c1 = 1/(1+math.exp(-dot1[i]))
            elif dot2[i]-dot1[i]>20:
                c1 = 0
                if dot2[i]<-20:
                    c2 = 0
                else:
                    c2 = 1/(1+math.exp(-dot2[i]))
            else:
                if dot1[i]<-20:
                    c1 = 0
                else:
                    c1 = 1/(1+math.exp(-dot1[i])+math.exp(dot2[i]-dot1[i]))

                if dot2[i]<-20:
                    c2 = 0
                else:
                    c2 = 1/(1+math.exp(-dot2[i])+math.exp(dot1[i]-dot2[i]))

            if y[i]==1:
                if c1>0:
                    l*=c1
                c1 = 1 - c1
                c2 = -c2

            elif y[i]==2:
                if c2>0:
                    l*=c2
                c1= -c1
                c2 = 1- c2
            else:
                c1 = -c1
                c2 = -c2
                l*= 1+c1+c2

            sum1 = sum1 + x[i]*c1
            sum2 = sum2 + x[i]*c2

        theta1 = theta1 + a*sum1.transpose()
        theta2 = theta2 + a*sum2.transpose()

    cor=0
    wro =0
    for i in range(len(x)):
        dot1 = np.dot(x,theta1)
        dot2 = np.dot(x,theta2)
        if (dot1[i]-dot2[i])>20:
            c2 = 0
            if (dot1[i]<-20):
                c1 = 0
            else:
                c1 = 1/(1+math.exp(-dot1[i]))
        elif dot2[i]-dot1[i]>20:
            c1 = 0
            if dot2[i]<-20:
                c2 = 0
            else:
                c2 = 1/(1+math.exp(-dot2[i]))
        else:
            if dot1[i]<-20:
                c1 = 0
            else:
                c1 = 1/(1+math.exp(-dot1[i])+math.exp(dot2[i]-dot1[i]))

            if dot2[i]<-20:
                c2 = 0
            else:
                c2 = 1/(1+math.exp(-dot2[i])+math.exp(dot1[i]-dot2[i]))

        c3 = 1-c1-c2
        if c1>c2 and c1>c3:
            b = 1
        elif c2>c1 and c2>c3:
            b = 2
        elif c3>c1 and c3>c2:
            b = 3
        if b==y[i]:
            cor +=1
        else:
            wro+=1
    if cor>=maxc:
        if cor==maxc:
            if l>maxl:
                maxj = j
                maxc = cor
                maxl = l
                m_theta1 = theta1
                m_theta2 = theta2
        else:
            maxj = j
            maxc = cor
            maxl = l
            m_theta1 = theta1
            m_theta2 = theta2


    # print ("\u03F4", "1 = ", theta1.transpose())
    # print ("\u03F4", "2 = ", theta2.transpose())
    print ("%2d/100 Completed" %(j))
    print ("Time Elapsed = ", time.time() - elapse)
print ("\n\n")
print ("Total Time Elapsed = ", time.time()-totaltime)
print ("Accuracy = %4f" %(cor/1.5))
print("Learning Parameter = %2d" %(maxj))
print ("Likelihood : %3f" %(maxl))
print ("\u03F4", "1 = ", m_theta1.transpose())
print ("\u03F4", "2 = ", m_theta2.transpose())