import numpy as np
def count_classes(data):
    counts={}
    for label in data:
        if label in counts:
            counts[label]+=1
        else:
            counts[label]=1
    return counts

def entropy(data):
    counts = count_classes(data)
    n = sum(counts.values())
    probabilities=[ count/n for count in counts.values() ]
    entropy= -np.sum([p * np.log2(p) for p in probabilities if p>0])
    return entropy

data = [
    [2.5, 1.0, "yes"],
    [3.0, 1.5, "no"],
    [2.7, 1.2, "yes"],
    [3.2, 1.7, "no"],
    [2.9, 1.3, "yes"],
    [3.5, 1.9, "no"],
    [2.8, 1.1, "yes"],
    [3.1, 1.6, "no"],
    [2.6, 1.0, "yes"]
]
labels=[row[-1] for row in data]
# data=pd.read_csv('data.csv')
entropy_val=entropy(labels)
print(f"Entropy: {entropy_val:.4f}")