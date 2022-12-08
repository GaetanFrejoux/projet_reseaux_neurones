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


#Result_OR = perceptron_simple(X, W_OR, 0)
#plot_with_class(X, W_OR, Result_OR, "1.1 - OR")
## 1.2 Widrow-hoff


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
        #plot_with_class(x, w, yd, "1.2 Widrow-Hoff - Epoch " + str(i + 1))
        print("Epoch ", i + 1, " : ", erreur[i]) # affichade de l'erreur
        if (erreur[i] == 0 or (i != 0 and (erreur[i - 1] - erreur[i] == 0))): break

    return w, erreur


# 1.2.2
Data = np.loadtxt(URL_P2_D1)
CLASSIF = [1]*25 + [-1]*25

#w1, erreur1 = apprentissage_widrow(Data.T, CLASSIF, 10, 10)
#print(w1)
#print(erreur1)
# 1.2.3
Data = np.loadtxt(URL_P2_D2)

#w2, erreur2 = apprentissage_widrow(Data.T, CLASSIF, 10, 25)
#print(w2)
#print(erreur2)

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


#print(multiperceptron(x, w1, w2))


# 1.3.2 *Programmation apprentissage multicouches*
ALPHA = 0.5
#     w
# x1 ----> y3
#    \  /    \
#     \/      \
#     /\       >----> yf
#    /  \     /
#   /    \   /
# x2-----> y4
#     w
#def multiperceptron_widrow(x, yd, epoch, batch_size):
#
#    def activation(x):
#        return 1 / (1 + np.exp(-x))  # sigmoid
#
#    w1 = np.random.randn(3, 2)
#    w2 = np.random.randn(3)
#    #w1 = np.array([[1, 1], [0.1, 0.4], [0.8, 0.6]])
#    #w2 = np.array([1, 0.3, 0.9])
#
#    erreur = []
#    for i in range(epoch):
#        w1_temp = w1
#        w2_temp = w2
#        erreur.append(0)
#        for j in range(len(x[0])):
#            y5 = multiperceptron(x[:,j], w1, w2)
#            r5 = (y5 - (y5 * y5)) * (yd[j] - y5)
#
#            # calcul de y3
#            y3 = perceptron_simple(x[:,j], w1[:,0], 1)
#            r3 = y3 * (1 - y3) * r5 * w2[1]
#
#            # calcul de y4
#            y4 = perceptron_simple(x[:,j], w1[:,1], 1)
#            r4 = y4 * (1 - y4) * r5 * w2[2]
#
#            # print all
#            #print("x: ", x)
#            #print("y5: ", y5)
#            #print("r5: ", r5)
#            #print("y3: ", y3)
#            #print("r3: ", r3)
#            #print("y4: ", y4)
#            #print("r4: ", r4)
#            w2_temp += ALPHA * r5 * np.array([1, y3, y4])
#            w1_temp[1,1] += ALPHA * r4 * x[0,j]
#            w1_temp[2,1] += ALPHA * r4 * x[1,j]
#            w1_temp[1,0] += ALPHA * r3 * x[0,j]
#            w1_temp[2,0] += ALPHA * r3 * x[1,j]
#            erreur[i] += (yd[j] - y5)**2
#
#            if (j % batch_size) == 0:
#                w1 = w1_temp
#                w2 = w2_temp
#
#
#        #plot_with_class(x.T, w1, yd, "1.3.2 - Epoch " + str(i + 1))
#        print("Epoch ", i + 1, " : ", erreur[i]) # affichade de l'erreur
#        if (erreur[i] == 0 or (i != 0 and (erreur[i - 1] - erreur[i] == 0))): break
#
#    return w1, w2, erreur

def multiperceptron_widrow(x, yd, epoch, batch_size):

    def activation(x):
        return 1 / (1 + np.exp(-x))  # sigmoid

    w1 = np.random.randn(3, 2)
    w2 = np.random.randn(3)
    #w1 = np.array([[1, 1], [0.1, 0.4], [0.8, 0.6]])
    #w2 = np.array([1, 0.3, 0.9])

    erreur = []
    for i in range(epoch):
        w1_temp = w1
        w2_temp = w2
        erreur.append(0)
        for j in range(len(x[0])):
            y5 = multiperceptron(x[:,j], w1, w2)
            r5 = - (yd[j] - y5) * (1 - (y5 * y5))
            # calcul de y3
            y3 = perceptron_simple(x[:,j], w1[:,0], 1)
            r3 = y3 * (1 - y3) * r5 * w2[1]
            # calcul de y4
            y4 = perceptron_simple(x[:,j], w1[:,1], 1)
            r4 = y4 * (1 - y4) * r5 * w2[2]
            # print all
            #print("x: ", x)
            #print("y5: ", y5)
            #print("r5: ", r5)
            #print("y3: ", y3)
            #print("r3: ", r3)
            #print("y4: ", y4)
            #print("r4: ", r4)
            w2_temp += ALPHA * r5 * np.array([1, y3, y4])
            w1_temp[1,1] += ALPHA * r4 * x[0,j]
            w1_temp[2,1] += ALPHA * r4 * x[1,j]
            w1_temp[1,0] += ALPHA * r3 * x[0,j]
            w1_temp[2,0] += ALPHA * r3 * x[1,j]
            erreur[i] += (yd[j] - y5)**2

            if (j % batch_size) == 0:
                w1 = w1_temp
                w2 = w2_temp


        #plot_with_class(x.T, w1, yd, "1.3.2 - Epoch " +
        print("Epoch ", i + 1, " : ", erreur[i]) # affic
        if (erreur[i] == 0 or (i != 0 and (erreur[i - 1] - erreur[i] == 0))): break

    return w1, w2, erreur
    

x = np.array([[0,1,0,1],[0,0,1,1]])
#x = np.array([[0.35], [0.9]]) 
yd = np.array([0, 1, 1, 0])
#yd = np.array([0.5])


w1,w2,erreur = multiperceptron_widrow(x, yd, 100, 2)
print(erreur)

print(multiperceptron(x[:,0], w1, w2))
print(multiperceptron(x[:,1], w1, w2))
print(multiperceptron(x[:,2], w1, w2))
print(multiperceptron(x[:,3], w1, w2))
