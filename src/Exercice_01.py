import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resources import (URL_P2_D1, URL_P2_D2, URL_SAVE_RES_E01)


ALPHA = 0.1

# 1.1

W_OR = np.array([-0.5, 1, 1])
W_AND = np.array([-1.5, 1, 1])
X = np.array([[0,0],[0,1],[1,0],[1,1]])

def perceptron_simple(x, w, active):
    seuil = w[0]
    dot = np.dot(x, w[1:])
    x = seuil + dot
    return np.sign(x) if (active == 0) else np.tanh(x)

active = 0

result = perceptron_simple(X, W_OR, active)
# if result[i] = -1 then CLASS 0 else if result[i] = 1 then CLASS 1
plt.scatter(X[:, 0], X[:, 1], c=result)

# draw the hyperplane
if active == 0:
    a0 = np.linspace(-1, 2)
    a1 = (W_OR[0] + a0*W_OR[1]) / (-W_OR[2])
    plt.plot(a0, a1, 'r-')
elif active == 1:
    plt.axhline(y=W_OR[0], color='red', linestyle='-')


plt.show()
plt.savefig(URL_SAVE_RES_E01 + "1.1.png")


def plot_with_class(X,w,c,title):
    x = np.linspace(-2,2,50)
    y = (w[0] + x*w[1]) / (-w[2])
    plt.title(title)
    plt.plot(x, y)
    plt.scatter(X[:,0],X[:,1],c=c)
    plt.grid()
    plt.show()

# 1.2 Widrow-hoff

def apprentissage_widrow(x, yd, epoch, batch_size):
    w = np.random.randn(3)
    erreur = []
    for i in range(epoch):
        e = 0
        for j in range(len(x)):
            result = perceptron_simple(x[j], w, 0)
            e += (yd[i] - result)**2
            
            if (j % batch_size) == 0:
                w += ALPHA * (yd[j]-result)* np.array([1,x[j,0],x[j,1]])
            
        #plot_with_class(x,w,yd,"Widrow-Hoff")
        
        erreur.append(e)
        if (e == 0):
            print("Epoch: ", i)
            break
    return w, erreur


# 1.2.2
Data = np.loadtxt(URL_P2_D1)
CLASSIF = [1]*25 + [2]*25

w, erreur = apprentissage_widrow(Data.T, CLASSIF, 10, 10)
print(w)
print(erreur)
# 1.2.3
Data = np.loadtxt(URL_P2_D2)


# 1.3
