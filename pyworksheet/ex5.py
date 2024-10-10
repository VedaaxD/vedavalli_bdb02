def power_raised_to_the_base(b,p):
    answer=b**p
    return answer
def main():
    b=int(input("Enter the number you want as base: "))
    p=int(input("Enter the number you want to have as the power of the base: "))
    print("The base",b,"when raised to the power",p,"gives",power_raised_to_the_base(b,p))
if __name__=="__main__":
    main()
