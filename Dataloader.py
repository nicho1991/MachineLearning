#module for ML

#basic useful imports 
# matplotlib inline
import numpy as np
import contextlib as ctxlib
import collections
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib
from sklearn.datasets import fetch_openml




#at this point sort of unknowns 12-02-2019
from math import inf, nan, fabs
from numpy import linalg

def dummyaddfunc(x, y):
    return x + y

def anotherdummyfunc(x,y):
    return x + y



#get and plot Make_moons dataset
def MOON_GetDataSet(n_samples):
    if n_samples < 0:
        n_samples = 100
    make_moon = make_moons(n_samples)
    return make_moon

def MOON_Plot(X, y):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()
   
X, y = MOON_GetDataSet(n_samples=200)
#with seperat trains data.
X_train, X_test, y_train, y_test = train_test_split(X, y)

def MOON_Plot_withtrain(X, x_color, y, y_color, title="My title", xlable="", ylabe=""):
    plt.scatter(X[:,0], X[:,1],c=y)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabe)
    plt.show()




# MNIST DATASET 
def MNIST_PlotDigit(data):
    some_digit_image = data.reshape(28, 28)
    # TODO: add plot functionality for a single digit...
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
    interpolation="nearest")
    plt.axis("off")
    plt.show()

def MNIST_GetDataSet():
    # fetch once , takes long

    print('started')
    X,y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True)
    print('loaded')
    return X,y
  


def IRIS_GetDataSet():
    iris = datasets.load_iris()
    X = iris.data[:,0:2]
    Y = iris.target

    
def IRIS_Feature_plot(k):
    size = len(k.data[0])
    plt.figure(figsize=(20,20))
    counter = 0;
    for z in range(0,size):
        for i in range(0,size):
            counter = counter + 1
            plt.subplot(4,4, counter)
            plt.scatter(k.data.T[i], k.data.T[z], c=k.target, s=50*petalWidth, alpha=0.8, cmap='viridis')
            plt.xlabel(k.feature_names[i])
            plt.ylabel(k.feature_names[z])    
                  
# TEST CODE:
print("ready to use")
#================================================================
#                       TEST
#================================================================
""" print(dummyaddfunc(2,2))
print(anotherdummyfunc("hello ", "World"))

# Test of make_moon data

print("X.shape=",X.shape,", y.shape=",y.shape)
MOON_Plot(X,y)

#test of make_moon data with train & test data
MOON_Plot_withtrain(X_train, 'ro',  y_train, 'bo', "Training data", "X_train_data", "y_train_data")
MOON_Plot_withtrain(X_test, 'ro', y_test, 'bo', "Test data", "X_test data", "y_test_data")


X, y = MNIST_GetDataSet()
print("X.shape=",X.shape, ", y.shape=",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000, shuffle=True)

print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
MNIST_PlotDigit(X_train[12])

#missing feature plotting.
plt.figure(figsize = (8,8))
plt.subplot(321)
plt.scatter(X[:,0], X[:,1], c=Y) """