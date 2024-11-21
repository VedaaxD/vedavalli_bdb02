# Check for the presence of an AvaII recognition site, which can have two different sequences: GGACC and GGTCC. Use regular expressions.
# dna = "ATCGCGAATTCAC"
# pattern = GGACC and GGTCC
import re
def check_recognition_site(dna_seq,pattern1,pattern2):
    pattern = f"({pattern1}|{pattern2})"
    match = re.search(pattern,dna_seq)
    if match:
        print(f"Recognition site is found in the position {match.start()} of the DNA sequence.")
    else:
        print(f"Match is not found")
check_recognition_site("ATCGCGAATTCAC","GGACC","GGTCC")
check_recognition_site("ATCGCGAATTCAC","GGACC","CGAAT")