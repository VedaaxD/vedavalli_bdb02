# Write  a lambda function to sort a list of strings by the last character
strings=['apple','parrot','brinjal','coffee','tuples','java','hello','lemon']
sorted_strings=sorted(strings,key=lambda word: word[-1])
#without lambda functions
print(sorted_strings)
# def last_char(word):
#     return word[-1]
# sorted_strings=sorted(strings,key=last_char)
# print(sorted_strings)

