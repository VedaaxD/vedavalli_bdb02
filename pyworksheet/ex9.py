def alt_char_of_str(s):
    if len(s)==0:
        print("Enter a valid input.")
        return
    else:
        print(s[::2])
def main():
    s=str(input("Enter a string: "))
    alt_char_of_str(s)
if __name__=="__main__":
    main()