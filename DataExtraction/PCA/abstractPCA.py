import numpy as np
#check PCA
k= 2
A = np.array([[1,2,3,4,5],[19,27,28,29,39]])
A = A.T

sigma = np.cov(A)
[U,S,V] = np.linalg.svd(sigma)
Ureduced_rows = U[0:k]
Ureduced_cols = U[:,0:k]
z_rows = Ureduced_rows.dot(A)
z_cols = (Ureduced_cols.T).dot(A)
Aapprox_rows = (Ureduced_rows.T).dot(z_rows)
Aapprox_cols = Ureduced_cols.dot(z_cols)

print(z_cols)
print(" ")
print(A)
print(" ")
print(Aapprox_rows)
print(" ")
print(Aapprox_cols)
