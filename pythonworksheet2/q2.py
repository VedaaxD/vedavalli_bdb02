# Write a function to find the minimum and maximum number in a list, L.
L=[23,-44,87,20,78,100,999]
def find_max(L):
    max=L[0]#initializing max as 0th index
    for i in L:
        if i>max :
            max=i
    return max
def find_min(L):
    min=float('inf')
    for i in L:
        if i<min:
            min=i
    return min
def main():
    max=find_max(L)
    print("The maximum number is",max)
    min=find_min(L)
    print("The minimum number is",min)
if __name__=="__main__":
    main()
