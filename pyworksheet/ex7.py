#Write a function to print individual digits of a number, N.
def individual_digits(a):
    s=str(a)
    for i in s:
        print(i)
def main():
    a=int(input("Enter a number:"))
    individual_digits(a)
if __name__=="__main__":
    main()