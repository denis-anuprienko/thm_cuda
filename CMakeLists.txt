cmake_minimum_required(VERSION 2.8)

project(thm_cuda LANGUAGES CXX CUDA)

enable_language(CUDA)

set(SOURCES main.cpp problem.cpp problem.cu)

#set(CMAKE_CUDA_FLAGS "-arch=sm_52")
set(CMAKE_CXX_FLAGS  "-O3")

add_executable(${PROJECT_NAME} ${SOURCES})
