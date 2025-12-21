cmake_minimum_required(VERSION 3.18)
project(smallpt_gpu LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86) # Adapt to target GPU if needed
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
# find_package(GLUT REQUIRED) # Replaced by Qt
find_package(CUDAToolkit REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

include_directories(include)
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIRS})
# include_directories(${GLUT_INCLUDE_DIR})

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Define sources (using NEW names as per plan)
set(SOURCES
    src/main.cu
    src/path_tracer_kernels.cu
    src/rng_kernels.cu
    src/display_functions.cpp
    src/main_window.cpp
    src/log.c
    src/cuda_utils.cu
    src/render_utils.cpp
)

# Create executable
add_executable(smallpt_gpu ${SOURCES})

# Link libraries
target_link_libraries(smallpt_gpu
    PRIVATE
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    # ${GLUT_LIBRARIES}
    Qt6::Widgets
    CUDA::cudart
    m # Math library
)

# Compiler options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(smallpt_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()
