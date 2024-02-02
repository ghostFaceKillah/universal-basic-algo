cmake_minimum_required(VERSION 3.26)
project(basic_algos)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(neural_nets neural_nets.cpp)
target_link_libraries(neural_nets Eigen3::Eigen)

add_executable(rrt rrt.cpp)
target_link_libraries(rrt Eigen3::Eigen ${OpenCV_LIBS})

add_executable(astar astar.cpp)
target_link_libraries(astar ${OpenCV_LIBS})
