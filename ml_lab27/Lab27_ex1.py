import numpy as np
#LINEAR ALGEBRA
#Can you tell whether the matrix, A = [9  -15;  -15  21] is positive definite?
#a matrix A is said to be +ve definite if it satisfies: xTAx > 0 for all non-zero vectors x
#(which means the matrix always produces a +ve no when used in quad form)- which doesn't flip the dir of any vector,
#all eigenvalues of A are +ve -> this is imp because cov matrices must be positive semi-definite/definite for invertibility
#all leading principal minors (top -left sq submatrices) shd have +ve determinants
#checking symmetry of the matrix
def is_symmetric(A):
    return np.array_equal(A, A.T)

def has_pos_eigenvalues(A):
    eigenvalues = np.linalg.eigvals(A) #np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0) #returns True only if all eigen values>

def has_pos_determinants(A): #All leading principal minors have +ve determinants
    n=A.shape[0]
    for k in range(1,n+1):
        minor=A[:k,:k] #takes the top-left k*k sub matrix like A[:1,:1], then A[:2,:2] and so on until A[:n,:n]
        det=np.linalg.det(minor)
        if det <=0:
            return False
    return True
def main():
    A=np.array([[9,-15],[-15,21]])
    if is_symmetric(A) and has_pos_eigenvalues(A) and has_pos_determinants(A):
        print(f"Matrix A is positive definite")
    else:
        print(f"Matrix A is not positive definite")
if __name__=='__main__':
    main()