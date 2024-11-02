class FormulaError(Exception):
    pass
def calculator():
    while True:
        try:
            calc=input("Enter the expression/press q for quit:").split()
            if calc[0].lower() == 'q':
                print("Exiting calculator.")
                break
            first_value=float(calc[0])
            operator=calc[1]
            third_value=float(calc[2])
            if len(calc)!= 3 :
                raise FormulaError("Input should have 3 elements.")
            if operator not in ('+', '-'):
                raise FormulaError("Only addition and subtraction operations can be performed.")
            if operator == '+' :
                result = first_value + third_value
                print(f"Result is: {result:.2f}")
            elif operator == '-':
                result = first_value - third_value
                print(f"Result is: {result:.2f}")
        except ValueError:
            print("Error. Enter valid input.")
        except Exception as e :
            print(f"{e} has occured.")
calculator()

