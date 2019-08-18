print("====================================MAQ DECOMPO ALGO======================================")

import numpy as np

def gaussian_elimination_with_pivot(A,b):
  n = len(A)
  M = A

  i = 0
  for x in A:
    x.append(b[i][0])
    i += 1
  
  for k in range(n):
    for i in range(k,n):
      if abs(M[i][k]) > abs(M[k][k]):
        M[k], M[i] = M[i],M[k]
      else:
        pass
    for j in range(k+1,n):
      q = float(M[j][k]) / M[k][k]
      for m in range(k, n+1):
        M[j][m] -=  q * M[k][m]
  x = [0 for i in range(n)]

  x[n-1] = float(M[n-1][n])/M[n-1][n-1]
  for i in range (n-1,-1,-1):
    z = 0
    for j in range(i+1,n):
      z = z  + float(M[i][j])*x[j]
    x[i] = float(M[i][n] - z)/M[i][i]
  
  with open("x.txt", "w") as file:
    for i in x:
        file.write(str(i))
        file.write("\n")


if __name__ == "__main__":
  A , b = [], []
  with open("b.txt") as file:
      for line in file:
          b.append([float(x) for x in line.replace('\n', '').split(" ")])
  with open("A.txt") as file:
      for line in file:
          a = list(filter(lambda x: x != '', [x for x in line.replace('\n', '').split(" ")]))
          A.append(list(map(lambda x : float(x), a)))

  gaussian_elimination_with_pivot(A,b)