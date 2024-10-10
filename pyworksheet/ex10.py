def cat_str(s1,s2):
    if len(s1) == 0 and len(s2) == 0:
        print("Invalid input.")
    else:
        s1+" "+s2
        print(s1+" "+s2)
def main():
    s1=str(input("Enter the first string: "))
    s2=str(input("Enter the second string: "))
    cat_str(s1,s2)
if __name__=="__main__":
    main()