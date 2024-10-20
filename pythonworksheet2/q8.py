# Write a program to extract words from a string list, L whose first character is k.
L=['kitten','calamari','Kimchi','Paella','Cappucino','kebab','Kothu Parotta','Biryani']
def extract_words(L):
    lookfor='k'
    words=[ i for i in L if i[0].lower()==lookfor]
    print("The words starting with k are:",words)
def main():
    extract_words(L)
if __name__=="__main__":
    main()