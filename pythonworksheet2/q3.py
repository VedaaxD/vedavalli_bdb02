# Write a program to find the even numbers in a list, L.
L=[27,87,64,92,100,33,84,22,56]
#using list comprehension
even_no=[ num for num in L if num%2==0 ]
print(even_no)