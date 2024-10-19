# Write a program to find the maximum and minimum value of a dictionary
from math import inf
numbers={
    'A':1,'B':27,'C':9,'D':43,'E':76
}
maximum_value=max(numbers.values())
print("Maximum value in the dictionary is:",maximum_value)
minimum_value=min(numbers.values())
print("Minimum value in the dictionary is:",minimum_value)
#finding through loop
def find_max_min():
    max,min=float(-inf),float(inf)
    for value in numbers.values():
        if value > max:
            max=value
        if value < min:
            min=value
    return max,min
def main():
    max,min=find_max_min()
    print(f"The maximum and minimum values are {max} and {min} respectively.")
if __name__=="__main__":
    main()


