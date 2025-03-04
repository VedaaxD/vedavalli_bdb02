#Implement information gain measures. The function should accept data points for parents, data points for both children
# and return an information gain value.
from entropy_measure import entropy
def weighted_entropy(left_child,right_child):
    n_total=len(left_child)+len(right_child)
    left_total=len(left_child)
    right_total=len(right_child)
    H_left=entropy(left_child)
    H_right=entropy(right_child)
    return (H_left*(left_total)/n_total) + (H_right*(right_total)/n_total)

def ig(parent,left_child,right_child):
    H_parent=entropy(parent)
    H_weighted=weighted_entropy(left_child,right_child)
    IG=H_parent-H_weighted
    return IG

parent = ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"]
left_child = ["yes", "yes", "yes", "yes"]
right_child = ["no", "no", "no", "no", "yes"]
# left_child = ["yes", "yes", "yes", "yes","yes"]
# right_child = ["no", "no", "no", "no"]
information_gain=ig(parent,left_child,right_child)
print(f"INFORMATION_GAIN: {information_gain:.4f}")
