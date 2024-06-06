cmake_minimum_required(VERSION 3.10)
project(testProjAsd)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(testProjAsd main.cpp)
target_link_libraries(testProjAsd ${OpenCV_LIBS})

