cmake_minimum_required(VERSION 2.8)
project(orb_matcher)
set(CMAKE_STATIC_LIBRARY_PREFIX "")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "-rdynamic")  
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /home/olorin/env/lib/python3.6/site-packages/numpy/core/include)

find_package( OpenCV REQUIRED )
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories( 
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS} )
set( outpath "${CMAKE_CURRENT_LIST_DIR}/build" )
add_library(orb_matcher SHARED orb_matcher.cpp src/ORB_matcher.cc)
target_link_libraries(
orb_matcher
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PYTHON_LIBRARIES}
)
set_target_properties(orb_matcher PROPERTIES PREFIX "")
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
)