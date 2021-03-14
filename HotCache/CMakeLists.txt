add_library(kvcache INTERFACE)
target_include_directories(kvcache INTERFACE include)
target_link_libraries(kvcache INTERFACE TBB::tbb model commonInterfacesCXX pthread)

add_executable(kvcache-bench benchmark/KVCache_benchmark.cu)
target_link_libraries(kvcache-bench kvcache)

add_executable(kvcache-tail-latency benchmark/KVCache_tail_latency.cu)
target_link_libraries(kvcache-tail-latency kvcache)