set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(kvstore INTERFACE)
target_include_directories(kvstore INTERFACE include)
target_link_libraries(kvstore INTERFACE pthread)
target_link_libraries(kvstore INTERFACE lslab)
target_link_libraries(kvstore INTERFACE kvcache)
target_link_libraries(kvstore INTERFACE tbb)
target_link_libraries(kvstore INTERFACE commonInterfacesCXX)
target_link_libraries(kvstore INTERFACE commonInterfacesCUDA)
target_link_libraries(kvstore INTERFACE canonicalstore)

add_executable(KVStoreClientTest test/KVStoreClientTest.cu)
target_link_libraries(KVStoreClientTest GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main kvstore)
add_test(KVStoreClientTest KVStoreClientTest)

add_executable(KVStoreCtxTest test/KVStoreCtxTest.cu)
target_link_libraries(KVStoreCtxTest GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main kvstore)
add_test(KVStoreCtxTest KVStoreCtxTest)

add_executable(KVStoreInternalClientTest test/KVStoreInternalClientTest.cu)
target_link_libraries(KVStoreInternalClientTest GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main kvstore)
add_test(KVStoreInternalClientTest KVStoreInternalClientTest)

add_executable(KVStoreTest test/KVStoreTest.cu)
target_link_libraries(KVStoreTest GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main kvstore)
add_test(KVStoreTest KVStoreTest)

add_executable(SimpleFutureTest test/SimpleFutureTest.cu)
target_link_libraries(SimpleFutureTest GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main kvstore)
add_test(SimpleFutureTest SimpleFutureTest)

add_executable(model_service_rate benchmark/model_service_rate.cu)
