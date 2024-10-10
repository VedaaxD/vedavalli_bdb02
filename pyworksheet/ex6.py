#check if a given number is prime or not
from math import sqrt
def is_prime(n):
    if n==0 or n==1:
        print( n," is neither prime nor composite.")
        return
    if n<0:
        print("Error. Enter only positive integer")
        return
    if n==2 :
        print("The number 2 is a prime number")
        return
    if n>2 :
        square_root = int(sqrt(n))
        for i in range(2,square_root+1 ):
            if n % i == 0 :
                print("The number", n, "is a composite number.")
                return
    print("The number",n,"is a prime number.")
def main():
    n=int(input("Enter any positive integer: "))
    is_prime(n)
if __name__=="__main__":
    main()