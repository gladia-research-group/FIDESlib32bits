cmake_minimum_required (VERSION 3.25.2)

# C++ Config

set (CMAKE_CXX_STANDARD 20)

# Project

project (simple LANGUAGES CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages

find_package (fideslib REQUIRED CONFIG)

# FIDESlib configuration.

if (fideslib_FOUND)
    message (STATUS "FOUND PACKAGE FIDESlib")
    message (STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message (STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message (STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message (STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
    message (STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
    message (STATUS "FIDESlib compile options: ${FIDESLIB_COMPILE_OPTIONS}")
else ()
    message (FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

set (CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

# Executables definition.

add_executable (00_basic_workflow)
target_sources(00_basic_workflow PRIVATE src/00_basic_workflow.cpp)
target_link_libraries (00_basic_workflow PRIVATE fideslib::fideslib)

add_executable (01_bootstrapping)
target_sources(01_bootstrapping PRIVATE src/01_bootstrapping.cpp)
target_link_libraries (01_bootstrapping PRIVATE fideslib::fideslib)

add_executable (02_polynomials)
target_sources(02_polynomials PRIVATE src/02_polynomials.cpp)
target_link_libraries (02_polynomials PRIVATE fideslib::fideslib)

add_executable (03_simd)
target_sources(03_simd PRIVATE src/03_simd.cpp)
target_link_libraries (03_simd PRIVATE fideslib::fideslib)

add_executable (04_optimizations)
target_sources(04_optimizations PRIVATE src/04_optimizations.cpp)
target_link_libraries (04_optimizations PRIVATE fideslib::fideslib)

add_executable (05_gaussian)
target_sources(05_gaussian PRIVATE src/05_gaussian.cpp)
target_link_libraries (05_gaussian PRIVATE fideslib::fideslib)