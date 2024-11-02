def logdec(func):
    def wrapper(*args,**kwargs):
        print(f"Calling {func.__name__} with args:{args},kwargs:{kwargs}")
        addition=func(*args,**kwargs)
        print(f"{func.__name__} returned:{addition}")
        return addition
    return wrapper
@logdec
def add(a,b):
    print("a+b is")
    return a+b
a,b=1,2
print("the sum of the 2 numbers are",add(a,b))
