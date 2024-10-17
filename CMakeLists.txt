cmake_minimum_required(VERSION 3.22)

project(vulkan-test)

set(CMAKE_CXX_STANDARD 17)

add_executable(app main.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(app PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(app PRIVATE ${Vulkan_LIBRARIES})

find_package(glfw3 CONFIG REQUIRED) # 追加
target_link_libraries(app PRIVATE glfw) # 追加


find_package(glm CONFIG REQUIRED)
target_link_libraries(app PRIVATE glm::glm)

set(MINIAUDIO_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
target_include_directories(app PRIVATE ${MINIAUDIO_INCLUDE_DIR})