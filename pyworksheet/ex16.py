def check_anagram(s1,s2):
    s1=s1.replace(" ","").lower()
    s2=s2.replace(" ","").lower()
    sorted_s1=sorted(s1)
    sorted_s2=sorted(s2)
    if sorted_s1 == sorted_s2 :
        print("Both words are anagrams of each other.")
    else :
        print("Both words are not anagrams of each other.")
def main():
    s1=str(input("Enter a string: "))
    s2=str(input("Enter a string: "))
    check_anagram(s1,s2)
if __name__=="__main__":
    main()