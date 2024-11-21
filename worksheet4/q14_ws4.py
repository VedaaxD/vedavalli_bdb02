#Download dna.txt from the usual code folder in my drive. The file contains a made up DNA sequence. Predict the fragment
# lengths that we will get if we digest the sequence with two made up restriction enzymes – AbcI, whose recognition site
# is ANT/AAT, and AbcII, whose recognition site is GCRW/TG. The forward slashes (/) in the recognition sites represent
# the place where the enzyme cuts the DNA.
import re
with open('C:\\Users\\vedav\\Downloads\\dna.txt','r') as file:
    dna_seq=file.read().strip()

AbcI_pattern=r'A.T(?=AAT)'
AbcII_pattern=r'GC[AG][AT](?=TG)'

AbcI_cut=[ match.start() + 5 for match in re.finditer(AbcI_pattern,dna_seq)]
AbcII_cut=[ match.start() + 3 for match in re.finditer(AbcII_pattern,dna_seq)]

cut_positions = sorted(set(AbcI_cut + AbcII_cut))
cut_positions = [0] + cut_positions + [len(dna_seq)]

fragment_lengths = [cut_positions[i + 1] - cut_positions[i] for i in range(len(cut_positions) - 1)]

print("Fragment lengths:", fragment_lengths)