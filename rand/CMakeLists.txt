add_library(rand SHARED src/zipf.cpp)
target_include_directories(rand PUBLIC include)

add_library(rand_static STATIC src/zipf.cpp)
target_include_directories(rand_static PUBLIC include)

enable_testing()
add_subdirectory(test)

get_target_property(PUBLIC_HEADER_DIR_RAND rand INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE PUBLIC_HEADERS_RAND ${PUBLIC_HEADER_DIR_RAND}/*)

set(DOXYGEN_INPUT_DIR "${DOXYGEN_INPUT_DIR} ${PUBLIC_HEADER_DIR_RAND}" PARENT_SCOPE)
set(DOXYGEN_PUBLIC_HEADERS "${DOXYGEN_PUBLIC_HEADERS};${PUBLIC_HEADERS_RAND}" PARENT_SCOPE)
