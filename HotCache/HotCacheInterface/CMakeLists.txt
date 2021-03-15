add_library(hotcache INTERFACE)
target_include_directories(hotcache INTERFACE include)
target_link_libraries(hotcache INTERFACE kvcache)
