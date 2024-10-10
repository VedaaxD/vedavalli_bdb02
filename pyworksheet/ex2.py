def dec_to_binary(n):
    if n == 0 :
        return 0
    elif n < 0:
        return "Error,enter a valid input."
    binary = ""
    while n > 0 :
        remainder = n % 2
        binary = str(n % 2) + binary
        n = n // 2
    return binary

def main():
    n=int(input("Enter any whole number: "))
    binary=dec_to_binary(n)
    print("The binary number for the given decimal number is" , binary)
if __name__=="__main__":
    main()