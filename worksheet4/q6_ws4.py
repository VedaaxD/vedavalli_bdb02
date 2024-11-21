#Check for the presence of a BisI restriction site using regular expression character groups:
# A character group is a pair of square brackets with a list of characters inside them.
# dna = "ATCGCGAATTCAC"
# pattern = GCNGC, where N represents any base, i.e. A, T, G, C
import re
def restriction_site(dna_seq,pattern):
    match = re.findall('GC[ATGC]GC',dna_seq)
    if match:
        print(f"Restriction site is found at the position {match} of the DNA sequence")
    else:
        print("Restriction site not found.")
restriction_site("ATCGCGAATTCAC",'GC[ATGC]GC')