project(bing)
cmake_minimum_required(VERSION 2.8)

catkin_package(CATKIN_DEPENDS INCLUDE_DIRS Objectness LibLinear)

add_subdirectory(LibLinear)
add_subdirectory(Objectness)
