cmake_minimum_required(VERSION 3.16)

project(ai_infra_learning)

set(CMAKE_CXX_STANDARD 17)

add_executable(
    request_manager
    cpp/day5_request_manager.cpp
)

add_executable(
    day6_request_manager
    cpp/day6_request_manager/main.cpp
    cpp/day6_request_manager/request_manager.cpp
)