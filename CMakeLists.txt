cmake_minimum_required(VERSION 3.10)
project(objectTracking)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(objectTracking parallel.cpp)
target_link_libraries(objectTracking ${OpenCV_LIBS})
#target_compile_options(my_target PRIVATE -ggdb -Wall)

