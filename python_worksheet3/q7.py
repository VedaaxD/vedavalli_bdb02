# Write a Python program to rearrange positive and negative numbers in a given array using Lambda.
a=[-2,8,80,-45,86,23,-76,-8,-34,100,64]
rearrange_a=sorted(a,key=lambda x: x>=0 )
print(rearrange_a)
#this method below was simply sorting in the ascending order
# rearrange_a=sorted(a)
# print(rearrange_a)