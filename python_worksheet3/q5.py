sentence="Hello, how are you?"
words=sentence.split()
reversed_sentence={word:word[::-1] for word in words}
print(reversed_sentence)