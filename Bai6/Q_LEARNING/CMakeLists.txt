add_library(Q_LEARNING q_learning.cpp)

target_include_directories(Q_LEARNING PUBLIC 
                                        "${PROJECT_SOURCE_DIR}/eigen/Eigen")
