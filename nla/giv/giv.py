import numpy as np
from math import hypot as norm2



print("====================================QR - Givense - DECOMPO ALGO======================================")

def G(A):
    
    (num_rows, num_cols) = np.shape(A)
    Q = np.identity(num_rows)
    R = np.copy(A)

    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    
    for (row, col) in zip(rows, cols): #
        if R[row, col] != 0:
          
            (c, s) = cal_CS(R[col, col], R[row, col])
            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)


def cal_CS(a, b):
  
    r = norm2(a, b)
    c = a/r #cos  = a/norm2(a,b)
    s = -b/r #sin = -b/norm2(a,b)

    return (c, s)
  
  
A , b = [], []
with open("b.txt") as file:
    for line in file:
        b.append([float(x) for x in line.replace('\n', '').split(" ")])
with open("A.txt") as file:
    for line in file:
        a = list(filter(lambda x: x != '', [x for x in line.replace('\n', '').split(" ")]))
        A.append(list(map(lambda x : float(x), a)))
  

(Q, R) = G(A)

b = np.asarray(b)
p = np.dot(Q.T, b)
print("x : \n")
print(np.dot(np.linalg.inv(R), p))

with open("Q.txt", "w") as file:
    for i in Q:
        file.write(str(i))
        file.write("\n")

with open("R.txt", "w") as file:
    for i in R:
        file.write(str(i))
        file.write("\n")