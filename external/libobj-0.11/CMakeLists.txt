FILE(GLOB LibObjSrc "obj/*.cpp")
FILE(GLOB LibObjHeaders "*.hpp" "obj/*.hpp")

set(TGT_NAME libobj)
add_library(${TGT_NAME} STATIC ${LibObjHeaders} ${LibObjSrc})
target_include_directories(${TGT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${TGT_NAME} PROPERTY FOLDER external)
