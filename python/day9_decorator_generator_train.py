# from functools import wraps

# def deco(func):
#     print("A")

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         print("B")
#         result = func(*args, **kwargs)
#         print("C")
#         return result

#     print("D")
#     return wrapper


# @deco
# def gen():
#     print("E")
#     yield 10
#     print("F")
#     yield 20
#     print("G")


# print("H")

# g = gen()

# print("I")

# x = next(g)
# print(x)

# print("J")

def deco(func):
    count = 0

    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        print("call:", count)
        return func(*args, **kwargs)

    return wrapper


@deco
def gen(start):
    x = start

    while x < start + 3:
        yield x
        x += 1


g1 = gen(10)
g2 = gen(20)

print(next(g1))
print(next(g1))
print(next(g2))
print(next(g1))
print(next(g2))