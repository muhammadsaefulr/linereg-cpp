cmake_minimum_required(VERSION 3.15)
project(linereg-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(linereg-cpp
    src/main.cpp
    src/linear_regression.cpp
)
