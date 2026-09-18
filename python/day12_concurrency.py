import time
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
import os

counter = 0

def change_counter() -> int:
    global counter
    counter += 1
    print(
        f"worker pid={os.getpid()}, "
        f"counter={counter}"
    )
    return counter


def process_task(name: str) -> str:
    pid = os.getpid()
    print(f"[process] {name}, pid={pid}")
    return f"{name} finished"


def task(name: str, seconds: int) -> str:
    print(
        f"[start] {name}, "
        f"thread={threading.current_thread().name}"
    )

    time.sleep(seconds)

    print(
        f"[end]   {name}, "
        f"thread={threading.current_thread().name}"
    )

    return f"{name} finished"


def main():
    start = time.perf_counter()

    # with ThreadPoolExecutor(max_workers=2) as pool:
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(task, "A", 3),
            pool.submit(task, "B", 1),
            pool.submit(task, "C", 1),
        ]

        print("all tasks submitted")

        for future in as_completed(futures):
            result = future.result()
            elapsed = time.perf_counter() - start
            print(f"[result] {result}, elapsed={elapsed:.2f}s")

    total = time.perf_counter() - start
    print(f"total={total:.2f}s")

    print("\n===== ProcessPoolExecutor =====")

    print("main pid =", os.getpid())

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(process_task, "A"),
            pool.submit(process_task, "B"),
            pool.submit(process_task, "C"),
            pool.submit(process_task, "D"),
        ]

        for future in as_completed(futures):
            print("[process result]", future.result())



    print("\n===== Process Memory =====")

    global counter
    counter = 0

    print(
        f"before: main pid={os.getpid()}, "
        f"counter={counter}"
    )

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(change_counter),
            pool.submit(change_counter),
        ]

        for future in futures:
            future.result()

    print(
        f"after: main pid={os.getpid()}, "
        f"counter={counter}"
    )


if __name__ == "__main__":
    main()