cmake_minimum_required(VERSION 2.8)

project(shale_2phase LANGUAGES CXX CUDA)

enable_language(CUDA)

set(SOURCES main.cpp problem.cpp problem.cu)

link_directories(${CUDA_PATH}/lib/x64)
link_libraries(cublas)

#set(CMAKE_CUDA_FLAGS "-arch=sm_52")
set(CMAKE_CXX_FLAGS  "-O3")

add_executable(${PROJECT_NAME} ${SOURCES})

#target_link_libraries(${PROJECT_NAME})
