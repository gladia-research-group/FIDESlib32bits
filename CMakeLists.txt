cmake_minimum_required(VERSION 3.25.2)
cmake_policy(VERSION 3.21.3...3.30)

############### Project configuration ###################

# Set target architectures.
if (NOT DEFINED FIDESLIB_ARCH)
	set (FIDESLIB_ARCH "70-real;70-virtual;80-real;86-real;89-real;90-real;90-virtual;100-real;120-real" CACHE STRING "Set the backend architecture target.")
endif()

# Set build type.
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set installation paths.
if (NOT DEFINED FIDESLIB_INSTALL_PREFIX)
	set (FIDESLIB_INSTALL_PREFIX "/usr/local" CACHE STRING "FIDESlib installation path.")
else ()
	get_filename_component(TMP_PATH_FIDES "${FIDESLIB_INSTALL_PREFIX}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
	message(STATUS "FIDESlib install path at: ${TMP_PATH_FIDES}")
	set (FIDESLIB_INSTALL_PREFIX "${TMP_PATH_FIDES}" CACHE STRING "FIDESlib installation path." FORCE)
endif()
if (NOT DEFINED OPENFHE_INSTALL_PREFIX)
	set (OPENFHE_INSTALL_PREFIX "/usr/local" CACHE STRING "OpenFHE installation path.")
else ()
	get_filename_component(TMP_PATH_OPENFHE "${OPENFHE_INSTALL_PREFIX}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
	message(STATUS "OpenFHE install path at: ${TMP_PATH_OPENFHE}")
	set (OPENFHE_INSTALL_PREFIX ${TMP_PATH_OPENFHE} CACHE STRING "OpenFHE installation path." FORCE)
endif()

# Option to auto-install patched OpenFHE.
option(FIDESLIB_INSTALL_OPENFHE "Set ON to install a patched version of OpenFHE." OFF)

# NATIVEINT backend width of the auto-installed OpenFHE. 32 additionally applies
# deps/openfhe-1.4.2-native32.patch (the CKKS encode fix the 32-bit build needs).
set(FIDESLIB_OPENFHE_NATIVE_SIZE "64" CACHE STRING
	"NATIVE_SIZE of the OpenFHE installed by FIDESLIB_INSTALL_OPENFHE (32 or 64).")
set_property(CACHE FIDESLIB_OPENFHE_NATIVE_SIZE PROPERTY STRINGS 32 64)

# Option to compile the benchmark suite.
option(FIDESLIB_COMPILE_BENCHMARKS "Set ON to compile the FIDESlib benchmark suite." ON)

# Option to compile the test suite.
option(FIDESLIB_COMPILE_TESTS "Set ON to compile the FIDESlib test suite." ON)

# Export compile commands for debugging.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

############### CPP ###################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# Compiler: default to the system/environment compiler.
# Pass -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ to use Clang explicitly.

############### CUDA Configuration ###################

# Set CUDA_PATH if not defined previously.
if (NOT DEFINED CUDA_PATH)
	set (CUDA_PATH "/usr/local/cuda")
endif ()

# Check if CUDA is installed.
if (NOT EXISTS "${CUDA_PATH}")
	message (FATAL_ERROR "CUDA not found at ${CUDA_PATH}. Please install CUDA.")
endif()

# Global language config.
set (CMAKE_CUDA_ARCHITECTURES "${FIDESLIB_ARCH}")
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_EXTENSIONS ON)
set (CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

################ Project Definition ###################

project(fideslib VERSION 1.0 DESCRIPTION "Homomorphic Encryption Library" LANGUAGES CXX CUDA)

################ Extra Flags ###################

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -rdc=true --diag-suppress=3133,611")

############### Utils ###################

include(FetchContent)
include(CMakePrintHelpers)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

############### Thread Building Blocks ###################

find_package(TBB)
if (TBB_FOUND)
	message(STATUS "TBB found: ${TBB_VERSION}")
else ()
	message(STATUS "TBB not found: building without TBB support")
endif ()

################ CUDA Libraries ###################

find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
	message(FATAL_ERROR "CUDA not found!")
else ()
	message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
	message(STATUS "CUDA include path: ${CUDAToolkit_INCLUDE_DIRS}")
endif ()

# Check for NCCL installation. Do not do anything more. It is managed by CUDA. This is just to
# enable conditional NCCL-dependent code.
find_library(nccl_FOUND NAMES nccl HINTS  /usr/local/cuda/lib64 /usr/lib /usr/local/lib)
if (nccl_FOUND)
  message(STATUS "Found NCCL: ${nccl_FOUND}")
  add_compile_definitions(MULTI_GPU NCCL RCCL)
  # Locate nccl.h: look one level up from the lib dir (e.g. .../nvidia/nccl/include)
  get_filename_component(_nccl_lib_dir "${nccl_FOUND}" DIRECTORY)
  link_directories("${_nccl_lib_dir}")
  find_path(NCCL_INCLUDE_DIR nccl.h
      HINTS "${_nccl_lib_dir}/../include" /usr/local/cuda/include /usr/include
      NO_DEFAULT_PATH)
  if (NCCL_INCLUDE_DIR)
      message(STATUS "Found nccl.h at: ${NCCL_INCLUDE_DIR}")
      include_directories(${NCCL_INCLUDE_DIR})
  else()
      message(WARNING "nccl.h not found next to NCCL library — add its directory to CMAKE_INCLUDE_PATH")
  endif()
else()
  message(STATUS "NCCL not found: building without NCCL support")
endif()

############### OpenFHE Configuration ###################

# Default runs will not update nor install OpenFHE. Specify manually to install.
if (FIDESLIB_INSTALL_OPENFHE)
	set(FIDESLIB_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
	set(FIDESLIB_OPENFHE_BUILD_SCRIPT "${FIDESLIB_DEPS_DIR}/build.sh")
	if (NOT EXISTS "${FIDESLIB_OPENFHE_BUILD_SCRIPT}")
		message(FATAL_ERROR "OpenFHE build script not found at ${FIDESLIB_OPENFHE_BUILD_SCRIPT}")
	endif()
	message(STATUS "Installing patched OpenFHE (NATIVE_SIZE=${FIDESLIB_OPENFHE_NATIVE_SIZE})")
	execute_process(COMMAND "${FIDESLIB_OPENFHE_BUILD_SCRIPT}" "${OPENFHE_INSTALL_PREFIX}"
					"${FIDESLIB_OPENFHE_NATIVE_SIZE}"
					WORKING_DIRECTORY "${FIDESLIB_DEPS_DIR}"
					COMMAND_ERROR_IS_FATAL ANY)
endif()

find_package(OpenFHE 1.4.2 CONFIG EXACT REQUIRED PATHS "${OPENFHE_INSTALL_PREFIX}")

option(BUILD_STATIC "Set to ON to include static versions of the library" ON)

if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# OpenMP: use -fopenmp=libomp for Clang, -fopenmp for GCC/other.
# Can be overridden: -DFIX_OMP_FLAGS="-fopenmp"
if (NOT DEFINED FIX_OMP_FLAGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set (FIX_OMP_FLAGS "-fopenmp=libomp")
    else()
        set (FIX_OMP_FLAGS "-fopenmp")
    endif()
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIX_OMP_FLAGS}")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${FIX_OMP_FLAGS}")

############### Optimization Flags ###################

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -Xcompiler -O0 -Xcompiler -g -G")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O2 -Xcompiler -g")
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
	set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Os")
endif ()

############### Configuration export ###################

set(CMAKE_INSTALL_PREFIX "${FIDESLIB_INSTALL_PREFIX}")

set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
set(INSTALL_BINARY_DIR "${CMAKE_INSTALL_BINDIR}")
set(INSTALL_DATA_DIR "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake")

############### FIDESlib Definition ###################

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api")

file(GLOB_RECURSE SOURCE_FILES_CUDA ${SOURCE_DIR}/*.cu)
file(GLOB_RECURSE SOURCE_FILES_CXX ${SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES_CUDA ${SOURCE_DIR}/*.cuh)
file(GLOB_RECURSE HEADER_FILES_CXX ${SOURCE_DIR}/*.hpp ${SOURCE_DIR}/*.h ${SOURCE_DIR}/*.inc)
set_source_files_properties(${SOURCE_FILES_CUDA} ${HEADER_FILES_CUDA} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${SOURCE_FILES_CXX} ${HEADER_FILES_CXX} PROPERTIES LANGUAGE CXX)

file(GLOB_RECURSE SOURCE_FILES_API ${API_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES_PUBLIC ${API_DIR}/*.hpp)
set_source_files_properties(${HEADER_FILES_PUBLIC} ${SOURCE_FILES_API} PROPERTIES LANGUAGE CXX)

add_library(fideslib STATIC)

# MGPU sources require NCCL; exclude them when building without multi-GPU support
if (NOT nccl_FOUND)
    list(FILTER SOURCE_FILES_CUDA EXCLUDE REGEX ".*MGPU.*")
    list(FILTER SOURCE_FILES_CXX  EXCLUDE REGEX ".*MGPU.*")
endif()

target_sources(fideslib PRIVATE ${SOURCE_FILES_CXX} ${SOURCE_FILES_CUDA} ${SOURCE_FILES_API})
target_sources (fideslib PRIVATE 
	FILE_SET private_headers
	TYPE HEADERS 
	BASE_DIRS "${SOURCE_DIR}"
	DESTINATION "${INSTALL_INCLUDE_DIR}"
	FILES ${HEADER_FILES_CXX} ${HEADER_FILES_CUDA}
)
target_sources (fideslib PUBLIC 
	FILE_SET public_headers
	TYPE HEADERS 
	BASE_DIRS "${API_DIR}"
	DESTINATION "${INSTALL_INCLUDE_DIR}"
	FILES ${HEADER_FILES_PUBLIC}
)

target_link_libraries(fideslib
	PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEpke_static.a"
	PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEbinfhe_static.a"
	PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEcore_static.a"
)

if (TBB_FOUND)
	message(STATUS "FIDESlib linked against TBB")
	target_link_libraries(fideslib PRIVATE TBB::tbb)
endif()

target_link_libraries(fideslib
	PRIVATE CUDA::cudart
	PRIVATE CUDA::cuda_driver
)

if (nccl_FOUND)
  message(STATUS "FIDESlib linked against NCCL")
  target_link_libraries(fideslib PRIVATE "${nccl_FOUND}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNCCL")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DNCCL")
endif()

set_target_properties(fideslib PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	PREFIX ""
)

############### GPU Checker ###################

add_executable (gpu-test EXCLUDE_FROM_ALL)

file (GLOB_RECURSE GPUTEST_SOURCES other/GPUTest.cpp)
set_source_files_properties(${GPUTEST_SOURCES} PROPERTIES LANGUAGE CUDA)

target_sources(gpu-test PRIVATE ${GPUTEST_SOURCES})
set_target_properties (gpu-test PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(gpu-test PRIVATE CUDA::cudart)

set_target_properties(gpu-test PROPERTIES 
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

############### Dummy executable ###################

add_executable (dummy)

file (GLOB_RECURSE DUMMY_SOURCES other/Dummy.cpp)
set_source_files_properties(${DUMMY_SOURCES} PROPERTIES LANGUAGE CXX)

target_sources(dummy PRIVATE ${DUMMY_SOURCES})
target_sources (dummy PUBLIC
	FILE_SET dependency_headers
	TYPE HEADERS
	BASE_DIRS ${SOURCE_DIR}
	FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
)

target_link_libraries (dummy 
	PRIVATE fideslib
)

set_target_properties (dummy PROPERTIES
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(dummy PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

############### Library Installation ###################

install(
	TARGETS fideslib
	EXPORT ${PROJECT_NAME}Targets
	LIBRARY DESTINATION "${INSTALL_LIBRARY_DIR}"
	ARCHIVE DESTINATION "${INSTALL_LIBRARY_DIR}"
	RUNTIME DESTINATION "${INSTALL_BINARY_DIR}"
	INCLUDES DESTINATION "${INSTALL_INCLUDE_DIR}"
	FILE_SET public_headers DESTINATION "${INSTALL_INCLUDE_DIR}"
	FILE_SET private_headers DESTINATION "${INSTALL_INCLUDE_DIR}"
)

install(
	EXPORT ${PROJECT_NAME}Targets
	FILE ${PROJECT_NAME}Targets.cmake
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION "${INSTALL_DATA_DIR}"
)

configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	INSTALL_DESTINATION "${INSTALL_DATA_DIR}"
)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
	@ONLY
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
	DESTINATION "${INSTALL_DATA_DIR}"
)

# Also install private headers so downstream targets can include implementation details when needed.
install(
	DIRECTORY "${SOURCE_DIR}/"
	DESTINATION "${INSTALL_INCLUDE_DIR}"
	FILES_MATCHING
		PATTERN "*.hpp"
		PATTERN "*.h"
		PATTERN "*.cuh"
		PATTERN "*.inc"
)


############### Test ###################

if (FIDESLIB_COMPILE_TESTS)

	set(INSTALL_GTEST OFF)

	FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0)
	FetchContent_MakeAvailable(googletest)
	include(GoogleTest)

	set(TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

	file(GLOB_RECURSE TEST_SOURCE ${TEST_SOURCE_DIR}/*.cpp ${TEST_SOURCE_DIR}/*.cu)
	set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)

	add_executable(fideslib-test)

	target_sources(fideslib-test PRIVATE ${TEST_SOURCE})

	target_sources (fideslib-test PRIVATE
		FILE_SET dependency_headers
		TYPE HEADERS
		BASE_DIRS ${SOURCE_DIR}
		FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
	)

	target_link_libraries(fideslib-test
		PRIVATE gtest
		PRIVATE gtest_main
		PRIVATE fideslib
	)

	set_target_properties(fideslib-test PROPERTIES 
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

endif()


############### Benchmark ###################

if (FIDESLIB_COMPILE_BENCHMARKS)

	set(BENCHMARK_ENABLE_INSTALL OFF)
	set(BENCHMARK_INSTALL_DOCS OFF)
	set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

	FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.8.3)
	FetchContent_MakeAvailable(benchmark)

	set(BENCH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bench)

	file(GLOB_RECURSE BENCH_SOURCE_FILES ${BENCH_SOURCE_DIR}/*.cu ${BENCH_SOURCE_DIR}/*.cpp)
	set_source_files_properties(${BENCH_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

	add_executable(fideslib-bench)

	target_sources(fideslib-bench PRIVATE ${BENCH_SOURCE_FILES})
	target_sources (fideslib-bench PUBLIC
		FILE_SET dependency_headers
		TYPE HEADERS
		BASE_DIRS ${SOURCE_DIR}
		FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
	)

	target_link_libraries(fideslib-bench
		PRIVATE benchmark::benchmark
		PRIVATE fideslib
	)

	set_target_properties (fideslib-bench PROPERTIES
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

endif()
