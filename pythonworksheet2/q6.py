# Write a program to extract elements of a list, if it occurs more than k times.
def extract_elements(k):
    L=['10','33','veda','90','8.4','lists','90','python','33','veda','90','veda','33','90','8.4','90']
    new_L=set([i for i in L if L.count(i)>k ])
    print("These are the elements occuring more than k times",new_L)
def main():
    k=int(input("Enter the value of k:" ))
    extract_elements(k)
if __name__=="__main__":
    main()