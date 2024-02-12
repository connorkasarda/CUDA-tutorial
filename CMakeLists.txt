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

# Adds application main file alone because CUDA
# Done because CUDA-specific symbols cannot be merged in same way as C++ functions
set(
    APP_SOURCE
    source/main.cu)

# Builds the CUDAtutorial library
add_library(CUDAtutorial_lib)
target_include_directories(CUDAtutorial_lib PUBLIC include)

# Add remaing CUDA tutorial source files to the library
target_sources(CUDAtutorial_lib PRIVATE
    source/kernal.cu
    source/threads.cu)

# Sets the CUDA property for separable compilation
set_target_properties(CUDAtutorial_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Builds the CUDAtutorial exectuable
add_executable(CUDAtutorial)
target_sources(CUDAtutorial PRIVATE ${APP_SOURCE})
target_include_directories(CUDAtutorial PUBLIC include)
target_link_libraries(CUDAtutorial PRIVATE CUDAtutorial_lib)
