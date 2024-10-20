# Write a program to subtract two matrices, m1 and m2, using a list of lists.
matrix_A=[[5,6,7],[9,8,6],[6,0,3]]
matrix_B=[[1,0,4],[7,3,2],[2,1,4]]
result_matrix=[]
def sub_matrix():
    result_matrix = []
    for i in range(len(matrix_A)):
        row=[]
        for j in range(len(matrix_A[i])):
            matrix=matrix_A[i][j]-matrix_B[i][j]
            row.append(matrix)
        result_matrix.append(row)
    return result_matrix
def main():
    result_matrix=sub_matrix()
    print("The resultant matrix is",result_matrix)
if __name__=="__main__":
    main()
#using list comprehensions
matrix_A = [[5, 6, 7], [9, 8, 6], [6, 0, 3]]
matrix_B = [[1, 0, 4], [7, 3, 2], [2, 1, 4]]
result_matrix=[[matrix_A[i][j]-matrix_B[i][j] for j in range(len(matrix_A[i]))] for i in range(len(matrix_A))]
print(result_matrix)

