print("====================================LU DECOMPO ALGO======================================")

import numpy as np

def lu(mat):

    n = len(mat)
    lower = [[0 for x in range(n)] for y in range(n)]
    upper = [[0 for x in range(n)] for y in range(n)]


    for i in range(n):
        # UPPER
        for k in range(i, n):
            # Summation of L(i, j) * U(j, k)
            sum = 0
            for j in range(i):
                sum += (lower[i][j] * upper[j][k])
            # Evaluating U(i, k)
            upper[i][k] = mat[i][k] - sum

        # LOWER
        for k in range(i, n):
            if (i == k):
                lower[i][i] = 1 # Diagonal as 1
            else:
                # Summation of L(k, j) * U(j, i)
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                #  Evaluating L(k, i)
                lower[k][i] = int((mat[k][i] - sum) / upper[i][i])

    return lower, upper


def solve(L, U, b):
    
    # ===========
    # Ax  = b
    # LUx = b
    # Ly  = b
    # Ux  = y
    # ===========


    n = len(L)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j]*y[j]
        y[i] = y[i]/L[i, i]
    y = np.asmatrix(y).T
    with open("y.txt", "w") as file:
        for i in y:
            file.write(str(i))
            file.write("\n")

    x = np.zeros(n)
    Uy = np.asarray(np.c_[U, np.asarray(y)],dtype=np.float64)

    x[n-1] =float(Uy[n-1][n])/Uy[n-1][n-1]
    for i in range (n-1,-1,-1):
        z = 0.0
        for j in range(i+1,n):
            z = z  + float(Uy[i][j])*x[j]
        x[i] = float(Uy[i][n] - z) / Uy[i][i]
    x = np.asmatrix(x).T

    return np.asarray(x)

A , b = [], []
with open("b.txt") as file:
    for line in file:
        b.append([float(x) for x in line.replace('\n', '').split(" ")])
with open("A.txt") as file:
    for line in file:
        a = list(filter(lambda x: x != '', [x for x in line.replace('\n', '').split(" ")]))
        A.append(list(map(lambda x : float(x), a)))

l, u  = lu(A)
L = np.asarray(l)
U = np.asarray(u)
b = np.asarray(b)


with open("L.txt", "w") as file:
    for i in L:
        file.write(str(i))
        file.write("\n")

with open("U.txt", "w") as file:
    for i in U:
        file.write(str(i))
        file.write("\n")



print("x : ", solve(L, U, b))