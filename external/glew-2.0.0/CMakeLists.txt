#
# glew-cmake project
#
# Provides a cmake build system for the GLEW project, and offers binary releases
# across several different platforms and compilers
#

# We require a minimum of V2.8 of CMake

message(${CMAKE_CURRENT_BINARY_DIR})

set(LIBBASENAME "glew")
# Project name (Will determine the output name)
project( ${LIBBASENAME})

# Setup the version numbers for the release we're currently at
set( LIB_VMAJOR 2 )
set( LIB_VMINOR 0 )
set( LIB_VPATCH 0 )


# Determine the pointer size to know what architecture we're targetting
if( ${CMAKE_SIZEOF_VOID_P} EQUAL 4 )
    set( PSIZE 32 )
elseif( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
    set( PSIZE 64 )
else()
    message( FATAL_ERROR "Data width could not be determined!" )
endif()

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( DBG "-debug" )
endif()

if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
    set( COMPILER "-gcc" )
elseif( MSVC )
    set( COMPILER "-vc" )

    # With MSVC we should normally generate a more specific compiler version
    # which includes information about the revision of the compiler. This
    # revision information lets us know which of the MSVCR libraries is depended
    # upon in our build.
    #
    # However, GLEW doesn't use the CRT so there is no need to differentiate
    # between different CRT versions.

endif()

if( WIN32 )
    list( APPEND MLIBS opengl32 )
    list( APPEND MLIBS gdi32 )
    list( APPEND MLIBS user32 )
    list( APPEND MLIBS kernel32 )
endif()

if( COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
    set( CMAKE_C_FLAGS_RELEASE "-Wall -O2" )
    set( CMAKE_C_FLAGS_DEBUG "-Wall -O0 -g" )
elseif( MSVC )
    set( CMAKE_C_FLAGS_RELEASE "/O2" )
    set( CMAKE_C_FLAGS_DEBUG "/Zi /O0" )
endif()

# Build the library name from the project and databus width (pointer size)
set( LIBNAME "${LIBBASENAME}${PSIZE}" )

# Include directories for all targets
include_directories( "${PROJECT_SOURCE_DIR}/include" )

# C Pre-processor definitions for all targets
add_definitions( -DGLEW_BUILD )
add_definitions( -DGLEW_NO_GLU )

# The shared GLEW library
add_library( ${LIBNAME} SHARED
    "${PROJECT_SOURCE_DIR}/src/glew.c"
    "${PROJECT_SOURCE_DIR}/build/glew.rc" 
)
set(GLEW_LIBRARY ${LIBNAME} PARENT_SCOPE)

target_link_libraries( ${LIBNAME} PUBLIC
    ${MLIBS} )
