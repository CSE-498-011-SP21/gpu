cmake_minimum_required(VERSION 3.18)

project(KVCG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)
message("USING ${CMAKE_TOOLCHAIN_FILE}")

find_package(Doxygen)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(GTest CONFIG REQUIRED)

enable_testing()

add_subdirectory(rand)

add_subdirectory(commonInterfaces)

add_subdirectory(Model)

add_subdirectory(HotCache)

add_subdirectory(CanonicalStore)

add_subdirectory(KVStore)

add_subdirectory(benchmark)

add_subdirectory(docs)

set(KVGPU_TARGETLIST ${KVGPU_TARGETLIST} kvstore rand)

install(TARGETS ${KVGPU_TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

set(CPACK_PACKAGE_VENDOR "dePaul Miller et. al.")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "dePaul Miller")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)
