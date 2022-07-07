import numpy as np



def Task1():
    A = np.random.randint(2, 8, (7, 7))
    print("A")
    print(A)
    U = np.zeros((7, 7), float)
    L = np.identity(7, float)
    result = 1
    for i in range(7):
        for j in range(7):
            if i <= j:
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

    for i in range(7):
        for j in range(7):
            if i==j: result=result*U[i, j]

    print("Определитель матрицы А", result)
    print()
    return A

def Task2(A):
    Q = np.zeros((7, 7), float)
    Q[:, 0] = A[:, 0]
    C = np.zeros(7, float)
    C[0] = np.dot(A[:, 1], Q[:, 0]) / np.dot(Q[:, 0], Q[:, 0])
    tmp = np.dot(C[0], Q[:, 0])
    Q[:, 1] = A[:, 1] - tmp

    for i in range(2, 7):
        for j in range(i):
            C[j] = np.dot(A[:, i], Q[:, j]) / np.dot(Q[:, j], Q[:, j])
        tmp = np.dot(C[0], Q[:, 0])
        for j in range(1, i):
            tmp += np.dot(C[j], Q[:, j])
        Q[:, i] = A[:, i] - tmp

    for i in range(7):
        norm = 0
        for j in range(7):
            norm += Q[j, i] * Q[j, i]
        norm = np.sqrt(norm)
        for j in range(7):
            Q[j, i] = Q[j, i] / norm

    Q1, R1 = np.linalg.qr(A)

    print("Q:\n", Q)
    print("np.linalg.qr")
    print("Q:\n", Q1)

    tmp = Q.transpose()
    R = np.dot(tmp, A)
    print("R:\n", R)
    print("np.linalg.qr")
    print("R:\n", R1)

    print("QR:\n", np.dot(Q, R))

def Task3():

    A = np.array([[3.6, 1.8, -4.7], [2.7, -3.6, 1.9], [1.5, 4.5, 3.3]])
    B = np.array([3.8, 0.4, -1.6])

    e = 0.001
    for i in range(3):
        A[2, i] -= A[0, i]

    C = np.zeros_like(A)
    F = np.zeros_like(B)
    n, m = A.shape
    for i in range(n):
        F[i] = B[i] / A[i, i]
        for j in range(m):
            if i != j:
                C[i, j] = -A[i, j] / A[i, i]

    print(f'A:\n {A}\n\n B:\n {B}\n\n C:\n {C}\n\n F:\n {F}\n\n')

    prevX = np.zeros_like(B)
    X = C.dot(prevX) + F
    iter = 1
    while (abs(X - prevX).max() > e):
        iter += 1
        prevX = X
        X = C.dot(prevX) + F

    print(
        f'X(Методом простых итераций):\n {np.round(X, 3)}\n\n Количество итераций:\n {iter}\n\n Проверка:\n {np.round(np.linalg.solve(A, B), 3)}\n\n ')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A = Task1()
    Task2(A)
    Task3()