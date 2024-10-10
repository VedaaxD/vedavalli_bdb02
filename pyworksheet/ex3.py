#counting no of 1s in the binary representation of a decimal number
#here we are using built in function bin() to convert the decimal number
def count1(n):
    binary_number=bin(n)[2:]
    N=binary_number
    #initializing count as 0
    count=0
    for i in binary_number:
        if i == '1' :
            count=count+1
    return count
def main():
    n=int(input("Enter a decimal number: "))
    answer=count1(n)
    print("The no of 1s in the binary representation of the decimal number is:",answer)
if __name__=="__main__":
    main()