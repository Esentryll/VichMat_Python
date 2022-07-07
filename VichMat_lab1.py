import warnings
warnings.filterwarnings("ignore", category=Warning)

import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt


def Task1():
    print()
    print("Задание 1")
    print("1.1. Создать матрицу 5x5 из единиц")
    A = np.ones((5, 5))
    print(A)
    print("1.2. Создать единичную матрицу 50х50")
    B = np.eye(50, 50)
    print(B)
    print()


def Task2():
    print()
    print("Задание 2")
    print("Вычислить определитель")
    matrix1 = np.matrix('3, -1, 2, 3, 2; 1, 2, -3, 3, 4; 2, -3, 4, 2, 1; 3, 0, 0, 5, 0; 2, 0, 0, 4, 0')
    print(matrix1)
    m = (int)(np.linalg.det(matrix1))
    print("Ответ:", m)
    print()


def Task3():
    print()
    print("Задание 3")
    print("3.1. Создать случайную матрицу A из целых чисел из отрезка [0,5] размера 4x4")
    A = np.random.random_integers(0, 5, (4, 4))
    print(A)
    print("3.2. Создать вектор-столбец B подходдящего размера")
    B = np.random.random_integers(0, 5, 4)
    print(B)
    print("3.3. Решить систему AX = B")
    print(A, '*X = ', B)
    a = np.linalg.inv(A)
    print("Mатрица,обратная A")
    print(a)
    result = np.dot(a, B)
    print("Ответ:")
    print('X=',result)
    print()


def Task4():
    print("Задание 4")
    print("Вычислите интеграл")
    f = lambda x: pow((math.cosh(3*x)),2)
    result = integrate.quad(f, 0, 1 / 3)
    print(result)
    print()

def Task5():
    print("Задание 5")
    print("Вычислите интеграл")
    f = lambda x,y,z: x+y+z
    g = lambda x,y: 1-x-y
    h = lambda x: 1-x
    result1 = integrate.tplquad(f, 0, 1, 0, h, 0, g)
    print(result1)
    print()

def Task6():
    print("Задание 6")
    plt.figure(figsize=(8, 5), dpi=80)
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
    C, L = 2*np.cos(X-np.pi/4), X+3

    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="y = 2*cos(x-pi/4)")
    plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="y = x+3")

    plt.xlim(X.min() * 1.1, X.max() * 1.1)

    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+3 \pi/2$',
                r'$+ 2\pi$'])

    plt.ylim(C.min() * 1.1, C.max() * 1.1)
    plt.yticks([-2, -1, +1, +2],
               [r'$-2$', r'$-1$', r'$+1$', r'$+2$'])

    plt.legend(loc='upper left', frameon=False)
    plt.grid()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Вариант 9")
    Task1()
    Task2()
    Task3()
    Task4()
    Task5()
    Task6()

