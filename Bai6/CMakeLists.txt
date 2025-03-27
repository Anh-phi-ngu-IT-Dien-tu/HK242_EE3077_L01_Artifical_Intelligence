cmake_minimum_required(VERSION 3.10)

project(main VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(Q_LEARNING)

#create executable file (window: .exe)
add_executable(main main.cpp)

#get .h file
target_include_directories(main PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen/Eigen"
                            "${PROJECT_SOURCE_DIR}/Q_LEARNING")

#get library or build object like .o or .a
target_link_libraries(main PUBLIC 
                            Q_LEARNING)
