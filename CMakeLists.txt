# Main cmake build file for the CUDA tutorial

# Sets cmake version minimum requirement for Ubuntu
cmake_minimum_required(VERSION 3.22.1)

# Sets the cmake version if too low as a backup plan
if(${CMAKE_VERSION} VERSION_LESS 3.22.1)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Sets project details
project(
    CUDAtutorial
    VERSION 2024.2.11
    DESCRIPTION "Space for learning cu code"
    LANGUAGES CXX CUDA)

# Adds application cu source files
set(
    APP_SOURCES
    source/main.cu)

# Builds the CUDAtutorial library
add_library(CUDAtutorial_lib ${APP_SOURCES})
target_include_directories(CUDAtutorial_lib PUBLIC include)

# Builds the CUDAtutorial exectuable
add_executable(CUDAtutorial ${APP_SOURCES})
target_include_directories(CUDAtutorial PRIVATE include)
target_link_libraries(CUDAtutorial PRIVATE CUDAtutorial_lib)
