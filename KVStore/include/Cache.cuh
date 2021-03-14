//
// Created by depaulsmiller on 3/14/21.
//

#ifndef KVCG_CACHE_CUH
#define KVCG_CACHE_CUH

template<typename K, typename V>
class Cache {
public:
    typedef kvgpu::KVCache<K, V, 1000000, 8> type;
};

#endif //KVCG_CACHE_CUH
