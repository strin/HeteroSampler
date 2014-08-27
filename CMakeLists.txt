CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT (POSTag)

include_directories("inc")
include_directories("scilog/inc")
link_directories("scilog")
add_subdirectory("src")


IF("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
set(CMAKE_PREFIX_PATH /opt/local ${CMAKE_PREFIX_PATH})

ENDIF()

find_package (Threads)

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
  ADD_DEFINITIONS("-std=c++11" "-static" "-g" "-O0"  "-fPIC")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
	ADD_DEFINITIONS("-std=c++11" "-O0" "-g" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

ADD_LIBRARY(tag SHARED
  src/feature.cpp
  src/corpus.cpp
  src/corpus_ocr.cpp
  src/tag.cpp
  src/MarkovTree.cpp
  src/model.cpp
  src/baseline.cpp
  src/search.cpp
  src/objcokus.cpp
  src/policy.cpp
  src/ThreadPool.cpp
)

target_link_libraries(tag
  scilog
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

add_executable(pos main.cpp
)

target_link_libraries(pos
  scilog
  tag
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)


add_executable(ocr main_ocr.cpp
)

target_link_libraries(ocr
  scilog
  tag
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)
add_executable(policy test_policy.cpp
)

target_link_libraries(policy
  scilog
  tag
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

add_executable(test-thread test_thread.cpp
src/ThreadPool.cpp
src/objcokus.cpp
)

target_link_libraries(test-thread
  scilog
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

