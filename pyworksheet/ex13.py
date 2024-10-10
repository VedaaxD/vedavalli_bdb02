def replace_char(s):
    s_list=list(s)
    if len(s_list)==0:
        print("Enter a valid input.")
        return
    else :
        for i in range(len(s_list)) :
            if s_list[i] == 'c':
                s_list[i]='d'
    new_str=''.join(s_list)
    print(new_str)
def main():
    s=str(input("Enter a string: "))
    replace_char(s)
    # replace(s)
# def replace(s):
#     new_s=s.replace("c","d")
#     print(new_s)
if __name__=="__main__":
    main()
