FILE(GLOB ImGui *.cpp)
FILE(GLOB ImGuiHeaders *.h)

set(TGT_NAME imgui)
add_library(${TGT_NAME} STATIC ${ImGuiHeaders} ${ImGui})
target_include_directories(${TGT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${TGT_NAME} PROPERTY FOLDER external)
