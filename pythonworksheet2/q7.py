# Write a program to remove all occurrences of an element from a list,
L=['10','33','veda','90','8.4','lists','90','python','33','veda','90','veda','33','90','8.4','90']
def remove_all_occurences(e):
    new_L=[ i for i in L if i!=e ]
    return new_L
def main():
    print(L)
    e=str(input("Enter the element you want to remove from the above list:"))
    new_L=remove_all_occurences(e)
    print(new_L)
if __name__=="__main__":
    main()