#Take a DNA sequence and determine whether or not it contains any ambiguous bases – i.e. any bases that are not A, T, G or C.
# If there are ambiguous bases, print all ambiguous bases and their positions.
import re
from re import finditer
dna = "CGATNCGGAACGATC"
def find_ambiguous_bases(dna):
    pattern=r"[^ATGC]"
    ambiguous_bases=[(match.group(),match.start()) for match in finditer(pattern,dna)]
    if ambiguous_bases:
        for base, position in ambiguous_bases:
            print(f"Ambiguous bases:{base} ; Position:{position}")
    else:
        print("No ambiguous bases found.")
    non_ambiguous = [base for base in dna if base not in ambiguous_bases]
    print(f"non-ambiguous bases:","".join(non_ambiguous))
def main():
    find_ambiguous_bases(dna)
if __name__=="__main__":
    main()
