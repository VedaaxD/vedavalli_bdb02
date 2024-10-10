def trim_whitespace(s):
    if len(s)==0:
        print("It is a empty string. Provide valid input.")
        return
    if len(s) > 0:
        new_str=s.lstrip()
    return new_str
def main():
    s=str(input("Enter a string:"))
    new_str=trim_whitespace(s)
    print("The new string after removing the whitespace characters is:",new_str)
if __name__=="__main__":
    main()