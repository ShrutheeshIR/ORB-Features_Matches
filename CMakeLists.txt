cmake_minimum_required(VERSION 2.8)
project( Example )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CFLAGS} -g")
find_package( OpenCV REQUIRED )
find_package(Eigen3 3.1.0 REQUIRED)

include_directories( 
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR} )
if (OpenCV_VERSION_MAJOR VERSION_EQUAL 4)
add_executable( Example runner.cpp orb_extractor/src/ORB.cc orb_extractor/src/ORB.hpp orb_matcher/src/ORB_matcher.cc orb_matcher/src/ORB_matcher.hpp)
endif()
target_link_libraries( Example 
${OpenCV_LIBS}
${EIGEN3_LIBS}
)
list(APPEND thirdparty_libraries
        ${OpenCV_LIBRARIES}
)