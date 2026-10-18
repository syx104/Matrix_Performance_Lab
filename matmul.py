import numpy as np

def naive_matmul(a,b):
    if a.ndim!=2 or b.ndim!=2:
        raise ValueError("Unputs must be 2D matrices")

    if a.shape[1]!=b.shape[0]:
        raise ValueError("Matrix dimensions do not match")

    m,k=a.shape
    n=b.shape[1]
    c=np.zeros((m,n),dtype=np.float64)

    for i in range(m):
        for j in range(n):
            for p in range(k):
                c[i,j]+=float(a[i,p])*float(b[p,j])

    return c

if __name__=="__main__":
    A=np.array([[1,2],[3,4]],dtype=np.float32)
    B=np.array([[5,6],[7,8]],dtype=np.float32)

    C=naive_matmul(A,B)

    print("Our result:")
    print(C)
    print("NumPy result:")
    print(A@B)
    print("Correct:",np.allclose(C,A@B))