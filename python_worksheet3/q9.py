import time
def timer(func):
    def wrapper(*args,**kwargs):
        begin=time.time()
        func(*args,**kwargs)
        end=time.time()
        print(f"The function {func.__name__} took {end-begin:.2f} seconds")
    return wrapper
@timer
def add_sleep(x):
    time.sleep(x)
add_sleep(2)
