cmake_minimum_required(VERSION 2.8.3)
project(octomap_compare)

find_package(catkin_simple 0.1.0 REQUIRED )
find_package(OpenCV 2 REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME}
               src/octomap_compare.cc
               src/octomap_container.cc
               src/random_forest_classifier.cc
               src/file_writer.cc)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

cs_add_executable(octomap_compare_online_node src/octomap_compare_online_node.cc)
target_link_libraries(octomap_compare_online_node ${PROJECT_NAME})

#add_doxygen(NOT_AUTOMATIC)

cs_install()
cs_export()

#############
# QTCREATOR #
#############
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
