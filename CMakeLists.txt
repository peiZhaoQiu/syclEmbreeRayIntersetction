cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/latest/linux/bin/icx)
set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/latest/linux/bin/icpx)
project(EMBREESYCL)

option(ENABLE_DEBUG "Enable Valgrind-based tests" OFF)
if(ENABLE_DEBUG)
   message(STATUS "DEBUG FLAG IS ON")
#    set(CMAKE_EXE_LINKER_FLAGS "-g -O0")
   add_compile_options(-g -O0)
endif()

if(ENABLE_SYCL)
   message(STATUS "SYCL FLAG IS ON")
   add_compile_options(-fsycl)
endif()



set(embree_DIR ${CMAKE_SOURCE_DIR}/external/embree-4.2.0/lib/cmake/embree-4.2.0)

FIND_PACKAGE(embree 4 REQUIRED)

#include all the files in the src directory
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/Test.cpp ${SOURCES})
if(ENABLE_SYCL)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include  /opt/intel/oneapi/compiler/latest/linux/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC embree sycl)
else()
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC embree)
endif()

