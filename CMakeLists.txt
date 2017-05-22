cmake_minimum_required(VERSION 2.8)

project(motion_detection CXX)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
