org1 = ["ACGTTTCA", "AGGCCTTA", "AAAACCTG"]
org2 = ["AGCTTTGA", "GCCGGAAT", "GCTACTGA"]
threshold=0.25
def similarity(seq1,seq2):
    match_seq=sum(1 for a,b in zip(seq1,seq2) if a==b )
    length = min(len(seq1), len(seq2))
    return match_seq/length
similar_pairs=[(seq1,seq2) for seq1 in org1 for seq2 in org2 if similarity(seq1,seq2) > threshold ]
print("These are the similar pairs of genome sequences:")
for pair in similar_pairs:
    print(pair)
