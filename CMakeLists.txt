cmake_minimum_required(VERSION 3.10)
project(BinaryImageMesher)

# Set C++ Standard to 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Library Source
set(LIB_SOURCES BinaryImageMesher.cpp)

# Main Executable
add_executable(pixmesh main.cpp ${LIB_SOURCES})
target_link_libraries(pixmesh ${OpenCV_LIBS})

# Tests Executable
add_executable(run_tests tests.cpp ${LIB_SOURCES})
target_link_libraries(run_tests ${OpenCV_LIBS})
