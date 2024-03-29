
PROJECT(facedetection)

SET(execName facedetection)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -g")

find_package(OpenCV REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB SRC "*.cpp")
add_executable(${execName} ${SRC})

TARGET_LINK_LIBRARIES( ${execName} ${OpenCV_LIBS})
