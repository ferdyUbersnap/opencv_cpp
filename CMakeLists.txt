cmake_minimum_required(VERSION 3.5)
project(OpenCVcpp)

# find opencv package
find_package(OpenCV REQUIRED)

# include directory from opencv
include_directories(${OpenCV_INCLUDE_DIRS})

# create an executable
add_executable(OpenCVcpp main.cpp)

# link Opencv libraries
target_link_libraries(OpenCVcpp ${OpenCV_LIBS})