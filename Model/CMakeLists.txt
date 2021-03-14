add_library(model INTERFACE)
target_include_directories(model INTERFACE include)
target_link_libraries(model INTERFACE Boost::boost)