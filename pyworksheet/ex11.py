# Write a function to find the first occurrence of a character, c, in a string S
def first_occurence(S,C):
    #finding the index of the first occurence of a character
    index=S.find(C)
    return index
def main():
    S=str(input("Enter a string: "))
    C=(input("Enter the character: "))
    index=first_occurence(S,C)
    print("The index of the first occurence of the character is",index)
if __name__=="__main__":
    main()