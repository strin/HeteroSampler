CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT (AdaInfer)

include_directories("inc")
include_directories("scilog/inc")
# link_directories("scilog")
add_subdirectory("src")
add_subdirectory("scilog")


IF("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
set(CMAKE_PREFIX_PATH /opt/local ${CMAKE_PREFIX_PATH})

ENDIF()

# Thread.
find_package (Threads)

# Boost.
ADD_DEFINITIONS(-DBoost_USE_STATIC_LIBS=OFF)
set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED ON)
SET(BOOST_MIN_VERSION "1.55.0")
find_package(Boost COMPONENTS program_options REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

# Python.
find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

# HDF5.
message(STATUS "build with HDF5 support")
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
add_definitions(${HDF5_CPPFLAGS})
add_definitions(-DWITH_HDF5)
message("Libs of HDF5: " ${HDF5_LIBRARIES} )

include_directories(
  "inc"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)


message(STATUS ${LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  message("Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-static" "-O0" "-g" "-fPIC")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
        ADD_DEFINITIONS("-std=c++11" "-O0" "-g" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

ADD_LIBRARY(heterosampler SHARED
  src/feature.cpp
  src/corpus.cpp
  src/corpus_ocr.cpp
  src/tag.cpp
  src/MarkovTree.cpp
  src/model.cpp
  src/baseline.cpp
  src/objcokus.cpp
  src/policy.cpp
  src/ThreadPool.cpp
)

target_link_libraries(heterosampler
  scilog
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${HDF5_LIBRARIES}
)

add_executable(bin/tagging model_tagging.cpp
)

target_link_libraries(bin/tagging
  scilog
  heterosampler
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)


add_executable(bin/ocr model_ocr.cpp
)

add_executable(bin/ising model_ising.cpp
)

target_link_libraries(bin/ocr
  scilog
  heterosampler
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

target_link_libraries(bin/ising
  scilog
  heterosampler
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

add_executable(bin/policy test_policy.cpp
)

target_link_libraries(bin/policy
  scilog
  heterosampler
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${HDF5_LIBRARIES}
)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY check)

add_executable(check-opengm-chain sanity/check_opengm_chain.cpp
)

target_link_libraries(check-opengm-chain
  scilog
  heterosampler
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${HDF5_LIBRARIES}
)
