import numpy as np
import random


def Task1():
    print("Задание 1")
    A = -1 + 2 * np.random.rand(8, 8)
    print("Матрица А")
    print(A)
    str = A[0,:]
    stolb = A[:,7]
    print("1 строка", str)
    print("8 столбец", stolb)
    result = np.dot(str,stolb)
    print("Скалярное произведение 1 строки на 8 столбец")
    print(result)
    print()

def Task2():
    print("Задание 2")
    n = 3
    k = 4
    m = 5
    print("Матрица А")
    A = np.random.randint(-6, 6, (n, k))
    print(A)
    print("Матрица B")
    B = np.random.randint(-6, 6, (k, m))
    print(B)
    C = np.zeros((n, m), dtype=int)
    print("Результат скалярного алгоритма:")
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i, j] = C[i, j] + A[i, p] * B[p, j]
    print(C)
    D = np.zeros((n, m), dtype=int)
    print("Результат векторного алгоритма:")
    for i in range(n):
        for j in range(m):
            D[i, j] = np.dot(A[i, :], B[:, j])
    print(D)
    print("Результат функции np.dot():")
    print(np.dot(A,B))
    print()

def Task3():
    print("Задание 3")
    print("Матрица А")
    k=0
    A = np.eye(7)
    for i in range(7):
        for j in range(7):
            if j<i:
                k=k+1
                A[i,j]=k
    print(A)
    print("Матрица B")
    B = np.random.randint(1, 10, (7, 1))
    print(B)
    print("Решить AX=B")
    a = np.linalg.inv(A)
    print("Mатрица,обратная A")
    print(a)
    result = np.dot(a, B)
    print("Ответ:")
    print(result)
    print("Проверка:")
    print("Вектор B")
    result1 = np.dot(A, result)
    print(result1)
    print()

def Task4():
    print("Задание 4")
    n = 4
    A = np.array([[7,-1.8,1.9,-57.4], [1,-4.3,1.5,-1.7], [2,1.4,1.6,1.8], [1,-1.3,-4.1,5.2]])
    B = np.array([10,19,20,10])
    print(A)
    print(B)
    U = np.zeros((n, n), float)
    L = np.identity(n, float)
    for i in range(4):
        for j in range(4):
            if i <= j:
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

    print('U')
    print(U)
    print('L')
    print(L)

    l = np.linalg.inv(L)
    y = np.dot(l, B)
    u = np.linalg.inv(U)
    x = np.dot(u, y)
    print('X')
    print(x)
    print('AX = ')
    print(np.dot(A, x))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Task1()
    Task2()
    Task3()
    Task4()

