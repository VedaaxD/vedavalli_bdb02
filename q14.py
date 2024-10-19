# Remove all duplicate words from given sentence using a dictionary
sentence="When I had had enough, I told him that that was the last time I would do it."
print("The original sentence is:",sentence)
word=sentence.split()
uniq_word={} #creating empty dictionary
for w in word:
    uniq_word[w]='yes' #set any value for the keys which we should iterate
new_sentence=' '.join(uniq_word.keys())
print("The new sentence after removing duplicate words:",new_sentence)

