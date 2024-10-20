# Write a program to implement an insertion sort algorithm
import random
def insertion_sort(A):
    l=len(A)
    for j in range(2,l):
        key=A[j]
        i=j-1
        while i>=0 and A[i]>key :
            A[i+1]=A[i]
            i=i-1
        A[i+1]=key
    return A
def main():
    A=[random.randint(1,100) for _ in range(10)]
    print(A)
    print(insertion_sort(A))
if __name__=="__main__":
    main()