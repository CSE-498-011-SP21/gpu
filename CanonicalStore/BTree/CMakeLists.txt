#enable_testing()
#
#add_library(btree INTERFACE)
#target_include_directories(btree INTERFACE include)
#target_link_libraries(btree INTERFACE allocator)
#target_link_libraries(btree INTERFACE commonInterfacesCXX)
#target_link_libraries(btree INTERFACE commonInterfacesCUDA)
#
#get_target_property(PUBLIC_HEADER_DIR_LSLAB btree INTERFACE_INCLUDE_DIRECTORIES)
#file(GLOB_RECURSE PUBLIC_HEADERS_LSLAB ${PUBLIC_HEADER_DIR_LSLAB}/*)
#set(DOXYGEN_INPUT_DIR "${DOXYGEN_INPUT_DIR} ${PUBLIC_HEADER_DIR_LSLAB}" PARENT_SCOPE)
#set(DOXYGEN_PUBLIC_HEADERS "${DOXYGEN_PUBLIC_HEADERS};${PUBLIC_HEADERS_LSLAB}" PARENT_SCOPE)
#
#add_executable(btree_benchmark benchmark/benchmark.cu)
#target_link_libraries(btree_benchmark PRIVATE btree)
#
#add_executable(btree_memory_benchmark benchmark/benchmark_mem.cu)
#target_link_libraries(btree_memory_benchmark PRIVATE btree)
#
#add_executable(slabunified_test test/slabunified_test.cu)
#target_link_libraries(slabunified_test PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main btree)
#add_test(slabunified_test slabunified_test)