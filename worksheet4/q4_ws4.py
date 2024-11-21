# Test if a DNA sequence contains an EcoRI restriction site using regular expressions.
# dna = "ATCGCGAATTCAC"
# pattern = GAATTC
import re
def check_restrictionsite(dna_seq,pattern):
    match=re.search(pattern,dna_seq)
    if match:
        print(f"Found the restriction site in the DNA Sequence at this position {match.start()}")
        return True
    return False
check_restrictionsite("ATCGCGAATTCAC","GAATTC")