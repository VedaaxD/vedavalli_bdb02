# Write a program to print the duplicate elements in a list, L.
L=['10','33','veda','90','8.4','lists','90','python','33','veda']
duplicate=[i for i in L if L.count(i)>1 ]
print("The duplicates are",duplicate) #this returns all the repeated elements multiple times
L=['10','33','veda','90','8.4','lists','90','python','33','veda']
duplicate=set([ i for i in L if L.count(i)>1 ])
print("The duplicates are",duplicate) #set returns the duplicate elements only once
