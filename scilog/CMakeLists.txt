cmake_minimum_required(VERSION 2.6)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories("inc")

set(PROJECT_SRCS
  ${PROJECT_SOURCE_DIR}/log.cpp 
  testlog.cpp 
)
ADD_DEFINITIONS("-std=c++0x" "-Wall" "-g" "-O0" "-fPIC")
ADD_LIBRARY(scilog SHARED
${PROJECT_SRCS})

add_executable(testlog testlog.cpp)

target_link_libraries(testlog
scilog)
