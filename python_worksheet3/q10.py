def division(a,b):
    try:
        quotient=(a/b)
        print(f"The value after dividing a by b is {quotient}")
    except ZeroDivisionError:
        print("Error. I am sure you know that division by zero isn't valid.")
    except ValueError:
        print("Enter valid input which are integers.")
    except Exception as e:
        print(f"An unexpected error {e} has occured.")
    finally:
        print("The division operation is completed.")

def main():
    try:
        a=int(input("Enter a number a:"))
        b=int(input("Enter a number b:"))
        division(a,b)
    except ValueError: #this additional error check to find whether the input entered itself is wrong, before going to the division function
        print("Enter valid input which are integers.")
    except Exception as e:
        print(f"An unexpected error {e} has occurred while reading input.")
if __name__=="__main__":
    main()

