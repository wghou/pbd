FILE(GLOB RPlySrc *.c)
FILE(GLOB RPlyHeaders *.h)

set(TGT_NAME rply)
add_library(${TGT_NAME} STATIC ${RPlyHeaders} ${RPlySrc})
target_include_directories(${TGT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${TGT_NAME} PROPERTY FOLDER external)