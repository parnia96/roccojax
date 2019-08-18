from math import sqrt
import numpy as np

print("====================================QR - HouseHolder- DECOMPO ALGO======================================")

def cmp(a, b):
    return (a > b) - (a < b) 

def mult_matrix(M, N):                                                                                                                                                                                                                                                
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N] for row_m in M]

def trans_matrix(M):
    n = len(M)
    return [[ M[i][j] for i in range(n)] for j in range(n)]

def norm(x):
    return sqrt(sum([x_i**2 for x_i in x]))

def Q_i(Q_min, i, j, k):
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]

def householder(A):
    n = len(A)
    R = A
    Q = [[0.0] * n for i in range(n)]

    for k in range(n-1):                                                                    
        I = [[float(i == j) for i in range(n)] for j in range(n)]

        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -cmp(x[0],0) * norm(x) # alpha = -sign(x1) * norm(x)

        u = list(map(lambda p,q: p + alpha * q, x, e))
        norm_u = norm(u)
        v = list(map(lambda p: p/norm_u, u))

        Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in range(n-k)] for j in range(n-k) ]
        Q_t = [[ Q_i(Q_min,i,j,k) for i in range(n)] for j in range(n)]

        if k == 0:
            Q = Q_t
            R = mult_matrix(Q_t,A)
        else:
            Q = mult_matrix(Q_t,Q)
            R = mult_matrix(Q_t,R)

    return trans_matrix(Q), R

A , b = [], []
with open("b.txt") as file:
    for line in file:
        b.append([float(x) for x in line.replace('\n', '').split(" ")])
with open("A.txt") as file:
    for line in file:
        a = list(filter(lambda x: x != '', [x for x in line.replace('\n', '').split(" ")]))
        A.append(list(map(lambda x : float(x), a)))
        
Q, R = householder(A)

q = np.asarray(Q)
r = np.asarray(R)
b = np.asarray(b)
p = np.dot(q.T, b)
print("x : \n")
print(np.dot(np.linalg.inv(r), p))

with open("Q.txt", "w") as file:
    for i in q:
        file.write(str(i))
        file.write("\n")

with open("R.txt", "w") as file:
    for i in R:
        file.write(str(i))
        file.write("\n")