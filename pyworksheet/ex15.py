def main():
    sentence=str(input("Enter any sentence: "))
    word=str(input("Enter the word that you want to count: "))
    if len(sentence)==0:
        print("Enter a valid input.")
    if len(word)==0:
        print("Enter a valid input.")
    else:
        normalized_sentence=sentence.lower().split()
        normalized_word=word.lower()
        print(normalized_sentence.count(normalized_word))

if __name__=="__main__":
    main()