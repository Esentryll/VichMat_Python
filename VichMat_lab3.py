import numpy as np
import matplotlib.pyplot as plt
import sympy as sm

def newton(data_set, x_p, x_7):
    result = data_set[0]
    for i in range(1, len(data_set)):
        p = data_set[i]
        for j in range(i):
            p *= (x_p - x_7[j])
        result += p
    return result

def table(x_, y):
    quotient = [[0] * len(x_) for _ in range(len(x_))]
    for n_ in range(len(x_)):
        quotient[n_][0] = y[n_]
    for i in range(1, len(x_)):
        for j in range(i, len(x_)):
            quotient[j][i] = (quotient[j][i - 1] - quotient[j - 1][i - 1]) / (x_[j] - x_[j - i])
    return quotient


def get_corner(result):
    link = []
    for i in range(len(result)):
        link.append(result[i][i])
    return link

def lagranz():
    x1 = np.array([0.41, 0.46, 0.52, 0.6, 0.65, 0.72], dtype=float)
    y = np.array([2.57418, 2.32513, 2.09336, 1.86203, 1.74926, 1.62098], dtype=float)
    z = 0
    x = sm.Symbol('x')
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x1)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (x - x1[i])
                p2 = p2 * (x1[j] - x1[i])
        z = z + y[j] * p1 / p2
    z = sm.expand(z)
    return z

if __name__ == '__main__':
    x1 = np.array([0.616, 0.478, 0.665, 0.537, 0.673], dtype=float)
    print('Task1')
    Task1 = lagranz()
    print(Task1)
    sm.plot(Task1)
    x = sm.Symbol('x')
    for i in range(5):
        print('Значение в точке',x1[i],'=',Task1.subs(x,x1[i]))

    print()
    print('Task2')
    xt = np.array([0.1539, 0.2569, 0.14, 0.2665], dtype = float)
    x2 = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], dtype=float)
    y2 = np.array([4.4817,4.9530,5.4739,6.0496,6.6859,7.3891,8.1662,9.0250,9.9742,11.0232,12.1825,13.4637], dtype=float)
    middle = table(x2, y2)
    n = get_corner(middle)
    for i in range(4):
        print("Значение для", xt[i],":", newton(n,xt[i],x2))
