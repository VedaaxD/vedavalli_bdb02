
def function(number):
    sum_of_squares = 0  # initializing sum of numbers to 0
    for number in range(0,number+1):
        sum_of_squares+=number**2
    return sum_of_squares
def main():
    number=int(input("Enter any number: "))
    answer=function(number)
    print("The sum of squares is",answer)
if __name__=="__main__":
    main()