# Write a function to find the  highest frequency character in a string, S.
def high_frq(S):
    max_frq=0
    max_chars=[]
    for char in S:
        count=S.count(char)
        if count > max_frq :
            max_frq = count
            max_chars=[char]
        elif count == max_frq and char not in max_chars:
            max_chars.append(char)
    return max_frq, max_chars

def main():
    S=str(input("Enter a string: "))
    max_frq,max_chars=high_frq(S)
    print("The character(s) with a maximum frequency is/are",max_chars,"with the frequency of",max_frq, "times.")

if __name__=="__main__":
    main()