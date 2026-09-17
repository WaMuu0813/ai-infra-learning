print("top-level running")

print("__name__ =", __name__)


def hello():
    print("hello function")


if __name__ == "__main__":
    print("main block running")
    hello()