enable_testing()

add_library(lslab INTERFACE)
target_include_directories(lslab INTERFACE include)
target_link_libraries(lslab INTERFACE allocator)
target_link_libraries(lslab INTERFACE commonInterfacesCXX)

add_executable(lslab_benchmark benchmark/benchmark.cu)
target_link_libraries(lslab_benchmark PRIVATE lslab)

add_executable(lslab_memory_benchmark benchmark/benchmark_mem.cu)
target_link_libraries(lslab_memory_benchmark PRIVATE lslab)

add_executable(slabunified_test test/slabunified_test.cu)
target_link_libraries(slabunified_test PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main lslab)
add_test(slabunified_test slabunified_test)