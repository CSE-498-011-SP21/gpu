cmake_minimum_required(VERSION 3.18)

project(KVCG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)
message("USING ${CMAKE_TOOLCHAIN_FILE}")
include(FetchContent)

FetchContent_Declare(
        betterstdlib
        GIT_REPOSITORY git@github.com:depaulmillz/betterstdlib.git
        GIT_TAG master
)

FetchContent_MakeAvailable(betterstdlib)

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

enable_testing()

add_subdirectory(commonInterfaces)

add_subdirectory(Model)

add_subdirectory(HotCache)

add_subdirectory(CannonicalStore)

add_subdirectory(KVStore)

add_executable(kvcg service/server.cu)
target_link_libraries(kvcg PUBLIC lslab)
target_link_libraries(kvcg PUBLIC multithreading)
target_link_libraries(kvcg PUBLIC pthread)
target_link_libraries(kvcg PUBLIC kvstore)
target_link_libraries(kvcg PUBLIC tbbmalloc_proxy)
target_link_libraries(kvcg PUBLIC Boost::boost)

add_library(zipfianWorkload SHARED service/zipfianWorkload.cu)
target_link_libraries(zipfianWorkload PRIVATE kvstore)
target_link_libraries(zipfianWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(zipfianWorkload PRIVATE rand)
target_link_libraries(zipfianWorkload PRIVATE Boost::boost)

add_library(zipfianWorkloadSwitch SHARED service/zipfianWorkloadSwitch.cu)
target_link_libraries(zipfianWorkloadSwitch PRIVATE kvstore)
target_link_libraries(zipfianWorkloadSwitch PRIVATE tbbmalloc_proxy)
target_link_libraries(zipfianWorkloadSwitch PRIVATE rand)
target_link_libraries(zipfianWorkloadSwitch PRIVATE Boost::boost)

add_library(uniformWorkload SHARED service/uniformWorkload.cu)
target_link_libraries(uniformWorkload PRIVATE kvstore)
target_link_libraries(uniformWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(uniformWorkload PRIVATE rand)
target_link_libraries(uniformWorkload PRIVATE Boost::boost)

add_library(selfsimilarWorkload SHARED service/selfsimilarWorkload.cu)
target_link_libraries(selfsimilarWorkload PRIVATE kvstore)
target_link_libraries(selfsimilarWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(selfsimilarWorkload PRIVATE rand)
target_link_libraries(selfsimilarWorkload PRIVATE Boost::boost)

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
