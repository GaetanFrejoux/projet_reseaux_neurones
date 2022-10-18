import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resources import (URL_P2_D1, URL_P2_D2, URL_SAVE_RES_E01)


ALPHA = 0.1
RESULTS_LOCATION = URL_SAVE_RES_E01  # Simply the url where the figures are saved

# 1.1

W_OR = np.array([-0.5, 1, 1])
W_AND = np.array([-1.5, 1, 1])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def perceptron_simple(x, w, active):
    seuil = w[0]
    dot = np.dot(x, w[1:])
    x = seuil + dot
    return np.sign(x) if (active == 0) else np.tanh(x)


def plot_with_class(X, Weight, c, title, Save=False, Url=RESULTS_LOCATION):
    x = np.linspace(-1, 2)
    y = (Weight[0] + x*Weight[1]) / (-Weight[2])
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=c)
    plt.plot(x, y, 'r-')
    plt.grid()
    if (Save): plt.savefig(Url + title + '.png')
    else: plt.show()
    plt.close()


Result_OR = perceptron_simple(X, W_OR, 0)
plot_with_class(X, W_OR, Result_OR, "1.1 - OR")
# 1.2 Widrow-hoff


def apprentissage_widrow(x, yd, epoch, batch_size):
    w = np.random.randn(3)
    erreur = []
    for i in range(epoch):
        w_temp = w
        erreur.append(0)
        for j in range(len(x)):
            y = perceptron_simple(x[j], w, 1)  # with tanh
            r = - (yd[j] - y) * (1 - y * y)
            w_temp += ALPHA * r * np.array([1, x[j][0], x[j][1]])
            erreur[i] += r**2
            if (j % batch_size) == 0: w = w_temp
        plot_with_class(x, w, yd, "1.2 Widrow-Hoff - Epoch " + str(i + 1))

        if (erreur[i] == 0 or (i != 0 and (erreur[i - 1] - erreur[i] == 0))): break

    return w, erreur


# 1.2.2
Data = np.loadtxt(URL_P2_D1)
CLASSIF = [1]*25 + [-1]*25

w1, erreur1 = apprentissage_widrow(Data.T, CLASSIF, 10, 10)
print(w1)
print(erreur1)
# 1.2.3
Data = np.loadtxt(URL_P2_D2)

w2, erreur2 = apprentissage_widrow(Data.T, CLASSIF, 10, 25)
print(w2)
print(erreur2)

# 1.3 Perceptron multicouche

# 1.3.1 Mise en place d'un perceptron multicouche

def multiperceptron(x, w1, w2):
    def activation(x):
        return 1 / (1 + np.exp(-x))  # sigmoid

    u1 = np.dot(np.array([w1[0][0], w1[1, 0], w1[2, 0]]),
                np.array([1, x[0], x[1]]))
    y1 = activation(u1)

    u2 = np.dot(np.array([w1[0][1], w1[1, 1], w1[2, 1]]),
                np.array([1, x[0], x[1]]))
    y2 = activation(u2)

    uf = np.dot(w2, np.array([1, y1, y2]))
    yf = activation(uf)

    return yf


x = np.array([1, 1])
w1 = np.array([[-0.5, 0.5], [2.0, 0.5], [-1.0, 1.0]])
w2 = np.array([2.0, -1.0, 1.0])


print(multiperceptron(x, w1, w2))
