cmake_minimum_required(VERSION 3.16)

project(day11_threadpool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    threadpool_test
    src/main.cpp
    src/thread_pool.cpp
)

add_executable(
    future_demo
    src/future_demo.cpp
)

add_executable(
    packaged_task_demo
    src/packaged_task_demo.cpp
)