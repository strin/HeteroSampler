CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT (POSTag)

include_directories("inc")
add_subdirectory("src")


IF("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
set(CMAKE_PREFIX_PATH /opt/local ${CMAKE_PREFIX_PATH})

ENDIF()


ADD_DEFINITIONS(-DBoost_USE_STATIC_LIBS=ON)
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS program_options REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(
  "inc"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)


message(STATUS ${LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  message("Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
#  IF(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
#    MESSAGE(FATAL_ERROR "g++ version 4.8 is required!")
#  ENDIF()
  ADD_DEFINITIONS("-std=c++0x" "-Wall" "-g" "-O0" "-fPIC")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
	ADD_DEFINITIONS("-std=c++11" "-Wall" "-g" "-O3" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

add_executable(pos main.cpp
src/corpus.cpp
src/log.cpp
src/tag.cpp
src/MarkovTree.cpp
src/model.cpp
src/objcokus.cpp
)

add_executable(testlog src/log.cpp
testlog.cpp
)


target_link_libraries(testlog
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

target_link_libraries(pos
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

