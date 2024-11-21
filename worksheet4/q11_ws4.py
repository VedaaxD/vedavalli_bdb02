import re
dna = "ACTGCATTATATCGTACGAAATTATACGCGCG"
pattern=r"ATTATAT|AAATTATA"
dna_in_bold=re.findall(pattern,dna)
print(f"The bases in bold letters:{dna_in_bold}")