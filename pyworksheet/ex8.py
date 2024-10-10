def first_half_of_string(s):
    if len(s)==0:
        print("Enter a valid input")
        return
    else:
        middle=len(s)//2
        print(s[:middle])
def main():
    s=str(input("Enter a string: "))
    first_half_of_string(s)
if __name__=="__main__":
    main()