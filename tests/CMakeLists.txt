
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


# polar decomposition test
set (DECOMP_ROOT "${PROJ_TESTS_DIR}/decomp")
set (DECOMP_TEST_TARGET decomp_test)
file (GLOB decomp_src "${PROJ_TESTS_DIR}/decomp_test.cpp" ${PROJ_SRC_DIR}/numerics.h)
file (GLOB eigen_math "${DECOMP_ROOT}/MathFunctions.cpp" "${DECOMP_ROOT}/MathFunctions.h" "${DECOMP_ROOT}/Common.h")
add_executable (${DECOMP_TEST_TARGET} ${decomp_src} ${eigen_math})
target_include_directories( ${DECOMP_TEST_TARGET} PRIVATE ${EIGEN3_ROOT_DIR} ${PROJ_SRC_DIR})
source_group("main" FILES ${decomp_src} ${eigen_math})

# voxelization test
set (VOXELIZATION_TEST_TARGET "voxelization_test")
set (VOXELIZATION_ROOT "${PROJ_TESTS_DIR}/voxelization")
file (GLOB main_voxel_src "${PROJ_TESTS_DIR}/voxel_test.cpp" "${VOXELIZATION_ROOT}/voxel_render.*")
file (GLOB shaders_test "${PROJ_ROOT_DIR}/data/shaders_test/*.glsl")

add_executable (${VOXELIZATION_TEST_TARGET} ${main_voxel_src} ${shaders_test})
target_include_directories( ${VOXELIZATION_TEST_TARGET} PRIVATE ${PROJ_SRC_DIR})
target_link_libraries (${VOXELIZATION_TEST_TARGET} PRIVATE 
    pbd_common pbd_cuda_utils 
    glfw ${GLU_LIBRARIES} ${GLEW_LIBRARIES} imgui rply libobj
)

source_group("voxelization_test" FILES ${main_voxel_src})
source_group("shaders_test" FILES ${shaders_test})

set_target_properties(${DECOMP_TEST_TARGET} ${VOXELIZATION_TEST_TARGET} PROPERTIES LINKER_LANGUAGE CXX)
if(MSVC)
set_target_properties(${VOXELIZATION_TEST_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
endif()