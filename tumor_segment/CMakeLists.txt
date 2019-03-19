cmake_minimum_required(VERSION 2.8)
project( segment )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set (CMAKE_CXX_STANDARD 11)
add_executable( segment tumor_segment.cpp )
target_link_libraries( segment ${OpenCV_LIBS} )

