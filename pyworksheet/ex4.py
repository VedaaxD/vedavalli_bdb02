#converting binary to decimal
from unicodedata import decimal


def binary_to_decimal(N):
    binary_str=str(N)
    length=len(binary_str)
    decimal=0 #initializing decimal as 0
    result=""
    for i in range(length):
        power=length-i-1
        digit=(int(binary_str[i]))
        decimal= digit * ( 2 ** power )+decimal
        result=str(decimal)
    return result
def main():
    N=int(input("Enter binary number you want to convert to decimal: "))
    answer=binary_to_decimal(N)
    print("The decimal value of the binary representation is", answer)
if __name__=="__main__":
    main()