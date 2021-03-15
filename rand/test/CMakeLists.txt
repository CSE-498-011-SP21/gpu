add_executable(zipf_test zipf_test.cpp)
target_link_libraries(zipf_test PRIVATE rand)

add_test(zipf_test zipf_test)