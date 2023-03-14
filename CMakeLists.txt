cmake_minimum_required(VERSION 3.24)
project(lamport)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

add_executable(lamport  lamport2.cpp)
target_link_libraries(lamport PUBLIC MPI::MPI_CXX)