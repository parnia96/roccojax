print("====================================LUP DECOMPO ALGO======================================")

import numpy as np

def mult_matrix(M, N):                                                                                                                                                                                  
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N] for row_m in M]

def pivot_matrix(M):

    m = len(M)                                                                                                                                                                                      
    id_mat = [[float(i == j) for i in range(m)] for j in range(m)]                                                                                                                                                                                             
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(M[i][j]))
        if j != row:                                                                                                                                                                                                                          
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
    return id_mat

def lu_decomposition(A):

    n = len(A)                                                                                                                                                                                                                 
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]
                                                                                                                                                                                            
    P = pivot_matrix(A)
    PA = mult_matrix(P, A)
                                                                                                                                                                                                                     
    for j in range(n):

        L[j][j] = 1.0
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1
                                                                                                                                                               
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return (P, L, U)


def solve(P, L, U, b):

    # ==========
    # Ax   = b
    # LUPx = b
    # LUx  = Pb
    # Ly   = b
    # Ux   = y
    # ==========

    # print("========Ly=b========\n")
    bb = np.asmatrix(P) * np.asmatrix(b)
    with open("bb.txt", "w") as file:
        for i in bb:
            file.write(str(i))
            file.write("\n")

    n = len(L)

    y = np.zeros(n)
    for i in range(n):
        y[i] = bb[i]
        for j in range(i):
            y[i] -= L[i, j]*y[j]
        y[i] = y[i]/L[i, i]
    y = np.asmatrix(y).T
    with open("y.txt", "w") as file:
        for i in y:
            file.write(str(i))
            file.write("\n")

    # print("========Ux=y========\n")
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


P, L, U = lu_decomposition(A)
npl = np.asarray(L)
npu = np.asarray(U)
npp = np.asarray(P)
b = np.asarray(b)

with open("L.txt", "w") as file:
    for i in npl:
        file.write(str(i))
        file.write("\n")

with open("U.txt", "w") as file:
    for i in npu:
        file.write(str(i))
        file.write("\n")

with open("P.txt", "w") as file:
    for i in npp:
        file.write(str(i))
        file.write("\n")

print("x : \n", solve(npp, npl, npu, b))