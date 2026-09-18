import time


def preprocess():
    time.sleep(0.5)


def model_forward():
    time.sleep(1.5)


def postprocess():
    time.sleep(0.3)


def inference():
    preprocess()
    model_forward()
    postprocess()


def main():
    start = time.perf_counter()

    inference()

    end = time.perf_counter()

    print(f"total time = {end - start:.2f}s")


if __name__ == "__main__":
    main()