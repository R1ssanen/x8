cmake_minimum_required(VERSION 3.10)
project(s_skybox LANGUAGES C)

add_library(s_skybox SHARED main.c)

target_include_directories(s_skybox PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/
    ${CMAKE_SOURCE_DIR}/mastertools/include/
)

target_compile_options(s_skybox PRIVATE
    -Wall -Wextra -Wpedantic -march=native -mavx2 -mfma
    $<$<CONFIG:Debug>: -g -O0>
    $<$<CONFIG:Release>: -O3>
)

add_custom_command(TARGET s_skybox
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:s_skybox> ${CMAKE_CURRENT_SOURCE_DIR}
)
