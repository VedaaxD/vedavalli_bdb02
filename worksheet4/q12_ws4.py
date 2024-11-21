#Write a regular expression to split the DNA string wherever we see a base that isn't A, T, G or C. if the
# dna = "ACTNGCATRGCTACGTYACGATSCGAWTCG", the output should be ['ACT', 'GCAT', 'GCTACGT', 'ACGAT', 'CGA', 'TCG']
import re
from re import finditer
def split_dna_string(dna):
    pattern = r"[^ATGC]+"
    split_seq=re.split(pattern,dna)
    split_seq=[ seq for seq in split_seq if seq]
    print(f"Output: {split_seq}")
def main():
    dna="ACTNGCATRGCTACGTYACGATSCGAWTCG"
    split_dna_string(dna)
if __name__=="__main__":
    main()
