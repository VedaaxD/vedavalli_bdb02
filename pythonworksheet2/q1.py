# Write a function to find the sum and average of numbers in a list, L.
def sum_of_numbers(l):
    sum_of_num=sum(l)
    print("The sum of numbers are",sum_of_num)
    return sum_of_num
def avg_of_numbers(l,sum_of_num):
    length=len(l)
    average=sum_of_num/length
    print("The average of the numbers are",average)
    return average
def main():
    l=[1,2,3,4,5,6]
    sum_of_num=sum_of_numbers(l)
    avg_of_numbers(l,sum_of_num)
if __name__=="__main__":
    main()
#using list comprehension
# l=[1,2,3,4,5,6]
# sum=sum([l for l in l ])
# print(sum)