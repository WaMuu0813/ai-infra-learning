class CountIterator:
    def __init__(self, end):
        self.current = 0
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration

        value = self.current
        self.current += 1
        return value


it = CountIterator(3)

print(next(it))
print(next(it))
print(next(it))

print("\n--- generator ---")

def count():
    yield 0
    yield 1
    yield 2

g = count()

print("iter(g) is g:", iter(g) is g)

print(next(g))
print(next(g))

g2 = iter(g)

print("g2 is g:", g2 is g)
print(next(g2))