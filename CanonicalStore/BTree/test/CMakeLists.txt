add_executable(test_map test_map.cu)
add_executable(test_search test_map_search.cu)

add_test(BTREE_buildMap_test test_map)
add_test(BTREE_buildAndSearchMap_test test_search)