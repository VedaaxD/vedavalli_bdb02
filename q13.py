# Given a dictionary with a values list, extract the key whose value has the most unique values.
# Input : test_dict = {"Gfg" : [5, 7, 7, 7, 7], "is" : [6, 7, 7, 7], "Best" : [9, 9, 6, 5, 5]}
# Output : "Best"
# Explanation : 3 (max) unique elements, 9, 6, 5 of "Best".
test_dict = {"Gfg" : [5, 7, 7, 7, 7], "is" : [6, 7, 7, 7], "Best" : [9, 9, 6, 5, 5]}
set_a=[set(values) for values in test_dict.values()]
print("These are the unique elements in the values of the dictionary",set_a) #this prints unique values
lengths_of_sets=[len(l) for l in set_a]
print("The length of each set is",lengths_of_sets)
b=(max(lengths_of_sets)) #finding the maximum among the length
keys_with_max_uniq=list(test_dict.keys())[lengths_of_sets.index(b)]
print("The key with the maximum number of unique elements is",keys_with_max_uniq)