from distutils.log import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resources import URL_P2_D1, URL_P2_D2
ALPHA = 0.1
# 1.1


def perceptron_simple(x, w, active):
    seuil = w[0]
    dot = np.dot(x, w[1:])
    x = seuil + dot
    if active == 0:
        return np.sign(x)
    elif active == 1:
        return np.tanh(x)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
w = np.array([-0.5, 1, 1])
active = 0
result = perceptron_simple(x, w, active)
# if result[i] = -1 then CLASS 0 else if result[i] = 1 then CLASS 1
plt.scatter(x[:, 0], x[:, 1], c=result)

# draw the hyperplane
if active == 0:
    a0 = np.linspace(-1, 2, 2)
    a1 = (w[0] + a0*w[1]) / (-w[2])
    plt.plot(a0, a1, 'r-')
elif active == 1:
    plt.axhline(y=w[0], color='red', linestyle='-')


# plt.show()


# 1.2 Widrow-hoff

def apprentissage_widrow(x, yd, epoch, batch_size):
    w = np.random.randn(3)
    erreur = []
    for j in range(epoch):
        e = 0
        for i in range(batch_size):
            result = perceptron_simple(x[i], w, 0)
            e += 0.5*(yd[i]-result)**2
            w[0] = w[0] + ALPHA*(yd[i]-result)*1
            w[1] = w[1] + ALPHA*(yd[i]-result)*x[i][0]
            w[2] = w[2] + ALPHA*(yd[i]-result)*x[i][1]
        if (e == 0):
            print('Weights are found : ', w)
            break
        erreur.append(e)
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
