from functools import wraps


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[start] {func.__name__}")

        result = func(*args, **kwargs)

        print(f"[end] {func.__name__}")
        return result

    return wrapper


@log_call
def generate_batches(num_batches, batch_size):
    for i in range(num_batches):
        print(f"  generating batch {i}")
        yield [i] * batch_size


g = generate_batches(3, 4)

print("function name:", generate_batches.__name__)
print("generator created")

print("first:", next(g))

print("remaining:")
for batch in g:
    print("got:", batch)

print("try again:")
for batch in g:
    print("got:", batch)

print("done")
