import re
dna = "ATCGCGYAATTCAC"
def find_ambiguous_bases(dna):
    pattern=r"[^ATGC]"
    ambiguous_bases=re.findall(pattern,dna)
    if ambiguous_bases:
        print(f"Ambiguous bases found {ambiguous_bases}")
    else:
        print("No ambiguous bases found.")
    non_ambiguous = [base for base in dna if base not in ambiguous_bases]
    print(f"non-ambiguous bases:","".join(non_ambiguous))
def main():
    find_ambiguous_bases(dna)
if __name__=="__main__":
    main()
