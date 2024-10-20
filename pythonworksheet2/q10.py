# Write a program to sum all values of a dictionary.
from math import fsum
numbers={
    'A':1,'B':27,'C':9,'D':43,'E':76
}
total_sum=sum([ value for value in numbers.values() ])
print(total_sum)