cmake_minimum_required(VERSION 4.3)
project(Black_Scholes)

set(CMAKE_CXX_STANDARD 20)

add_executable(Black_Scholes main.cpp)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(Black_Scholes PRIVATE -O3 -march=native)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Black_Scholes PRIVATE OpenMP::OpenMP_CXX)
endif()
