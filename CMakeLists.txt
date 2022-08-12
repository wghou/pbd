cmake_minimum_required(VERSION 3.19)

project (pbd) 

set (PROJ_ROOT_DIR  ${CMAKE_SOURCE_DIR})
set (PROJ_SRC_DIR "${PROJ_ROOT_DIR}/src")
set (PROJ_TESTS_DIR "${PROJ_ROOT_DIR}/tests")
set (EXTERNAL_DIR   "${PROJ_ROOT_DIR}/external")
set (CMAKE_DIR   "${PROJ_ROOT_DIR}/cmake")

set (VOXEL_LIB_DIR "${PROJ_SRC_DIR}/voxelization_lib")
set (SDF_LIB_DIR "${PROJ_SRC_DIR}/sdf_lib")


set (GLFW_ROOT_DIR  "${EXTERNAL_DIR}/glfw-3.2.1")
set (GLEW_ROOT_DIR  "${EXTERNAL_DIR}/glew-2.0.0")
set (GLM_ROOT_DIR   "${EXTERNAL_DIR}/glm-0.9.8.4")
set (IMGUI_ROOT_DIR "${EXTERNAL_DIR}/imgui")
set (RPLY_ROOT_DIR  "${EXTERNAL_DIR}/rply-1.1.4")
set (LIBOBJ_ROOT_DIR  "${EXTERNAL_DIR}/libobj-0.11")
set (CUB_ROOT_DIR "${EXTERNAL_DIR}/cub-1.7.0")
set (EIGEN3_ROOT_DIR "${EXTERNAL_DIR}/eigen")

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# get rid of debug and release directory created by visual studio
if (WIN32)
    foreach (OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string (TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        string (TOLOWER ${OUTPUTCONFIG} output_dir_name)
        set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJ_ROOT_DIR}/${output_dir_name})
        set (CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJ_ROOT_DIR}/${output_dir_name})
        # set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJ_ROOT_DIR}/${output_dir_name})
    endforeach (OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)
else (WIN32)
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJ_ROOT_DIR}/bin)
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJ_ROOT_DIR}/bin)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJ_ROOT_DIR}/bin)
endif (WIN32)

include (CheckCXXCompilerFlag)
if (WIN32)
else (WIN32)
    set (EXTRA_COMPILE_FLAGS "-Wall")
    CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CXX_11)
    if (HAS_CXX_11)
        set (EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -std=c++11")
        message (STATUS "Enabling C++11 support")
    else (HAS_CXX_11)
        message(FATAL_ERROR "No C++11 support detected, build will fail.")
    endif ()
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${EXTRA_COMPILE_FLAGS}")
endif (WIN32)

get_filename_component(CC_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

# CUDA
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set (compute_capability_query_src "${CMAKE_DIR}/compute_capability_query.c")
    set (compute_capability_query_bin_name "compute_capability_query")
    if (WIN32)
        set (compute_capability_query_bin_name "${compute_capability_query_bin_name}.exe")
    endif ()
    set (compute_capability_query_bin "${PROJ_ROOT_DIR}/build/${compute_capability_query_bin_name}")

    message(STATUS "run: ${CUDAToolkit_NVCC_EXECUTABLE} -ccbin ${CC_BIN_DIR} -lcuda ${compute_capability_query_src} -o ${compute_capability_query_bin}")
    execute_process (COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ccbin ${CC_BIN_DIR} -lcuda ${compute_capability_query_src} -o ${compute_capability_query_bin})
    execute_process (COMMAND ${compute_capability_query_bin}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE PBD_CUDA_COMPUTE_CAPABILITY
    )

    if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
        message(FATAL_ERROR "cannot retrieve the compute capability of the current cuda device, error: ${CUDA_RETURN_CODE}")
        message(STATUS "PBD_CUDA_COMPUTE_CAPABILITY: ${PBD_CUDA_COMPUTE_CAPABILITY}")
    else ()
        message(STATUS "PBD_CUDA_COMPUTE_CAPABILITY: ${PBD_CUDA_COMPUTE_CAPABILITY}")
    endif()
else()
message(FATAL_ERROR "cuda not found")
endif()

set(BUILD_SHARED_LIBS OFF)

# glm
include_directories (${GLM_ROOT_DIR})

# imgui
add_subdirectory (${IMGUI_ROOT_DIR}) 

# rply for pre-processing .ply files
add_subdirectory (${RPLY_ROOT_DIR}) 

#libobj for pre-processing wavefront .obj files
add_subdirectory (${LIBOBJ_ROOT_DIR}) 

# glfw
set (GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
set (GLFW_BUILD_TESTS    OFF CACHE BOOL "Build the GLFW test programs")
set (GLFW_BUILD_DOCS     OFF CACHE BOOL "Build the GLFW documentation")
set (GLFW_INSTALL        OFF CACHE BOOL "Generate installation target")

add_subdirectory (${GLFW_ROOT_DIR})
include_directories (${GLFW_ROOT_DIR}/include)

# glew
add_subdirectory (${GLEW_ROOT_DIR})
include_directories (${GLEW_ROOT_DIR}/include)

file (GLOB cuda_sdf_src "${PROJ_SRC_DIR}/sdf_cuda.h" "${PROJ_SRC_DIR}/sdf_cuda.cu")
file (GLOB cuda_voxelization_src "${PROJ_SRC_DIR}/voxelization_cuda.h" "${PROJ_SRC_DIR}/voxelization_cuda.cu")
file (GLOB cuda_solver_src "${PROJ_SRC_DIR}/solver_cuda.h" "${PROJ_SRC_DIR}/solver_cuda.cu")
file (GLOB cuda_scene_src "${PROJ_SRC_DIR}/scene_cuda.h" "${PROJ_SRC_DIR}/scene_cuda.cu")
file (GLOB cuda_math "${PROJ_SRC_DIR}/numerics.h" "${PROJ_SRC_DIR}/cuda_math_helper.h")

add_library(pbd_cuda_utils STATIC ${cuda_math} ${cuda_sdf_src} ${cuda_voxelization_src} ${cuda_solver_src} ${cuda_scene_src})

set_target_properties(pbd_cuda_utils PROPERTIES 
    CUDA_ARCHITECTURES "${PBD_CUDA_COMPUTE_CAPABILITY}"     # defined in top level cmake list file
    CUDA_STANDARD 11
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(pbd_cuda_utils PUBLIC ${PROJ_SRC_DIR} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(pbd_cuda_utils PUBLIC CUDA::cudart_static)

file (GLOB cg_lib "${PROJ_SRC_DIR}/aabb.*" "${PROJ_SRC_DIR}/bvh.*" "${PROJ_SRC_DIR}/cg_math.*" "${PROJ_SRC_DIR}/mesh.*")
file (GLOB helpers "${PROJ_SRC_DIR}/gl_helper.*" "${PROJ_SRC_DIR}/cuda_helper.h" "${PROJ_SRC_DIR}/misc.*" "${PROJ_SRC_DIR}/gui.*")
file (GLOB solver_src "${PROJ_SRC_DIR}/solver.*")
file (GLOB scene_src "${PROJ_SRC_DIR}/scene.*" "${PROJ_SRC_DIR}/main.cpp" )
file (GLOB gl_shaders "${PROJ_ROOT_DIR}/data/shaders/*.glsl")

add_library(pbd_common STATIC ${cg_lib} ${helpers})
target_link_libraries(pbd_common PUBLIC rply libobj imgui)

add_executable (${PROJECT_NAME} ${solver_src} ${scene_src} ${gl_shaders})

source_group ("pbd_common" FILES ${cg_lib} ${helpers})
source_group ("cuda_utils" FILES ${cuda_math} ${cuda_sdf_src} ${cuda_voxelization_src} ${cuda_solver_src} ${cuda_scene_src})
source_group ("main_src" FILES ${solver_src} ${scene_src})
source_group ("gl_shaders" FILES ${gl_shaders})

set (GLEW_LIBRARIES "glew")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set (GLEW_LIBRARIES "${GLEW_LIBRARIES}64")
else ()
    set (GLEW_LIBRARIES "${GLEW_LIBRARIES}32")
endif ()

set (GLU_LIBRARIES "GLU")
if (WIN32)
        set (GLU_LIBRARIES "${GLU_LIBRARIES}32")
else()
    target_link_libraries (${PROJECT_NAME} GL)
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE 
    pbd_common pbd_cuda_utils glfw ${GLU_LIBRARIES} ${GLEW_LIBRARIES}
)

set_target_properties(glfw ${GLEW_LIBRARIES} PROPERTIES FOLDER external)

# tests
option (PBD_TESTS "Build pbd test programs" ON)
set (PBD_TESTS ON CACHE BOOL "Build pbd test programs")
if (PBD_TESTS)
    add_subdirectory(${PROJ_TESTS_DIR})
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "all_build")


set (startup_target "pbd")

if (MSVC)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${startup_target})
endif ()
