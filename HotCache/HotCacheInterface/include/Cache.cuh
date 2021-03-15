//
// Created by depaulsmiller on 3/14/21.
//

#ifndef KVCG_CACHE_CUH
#define KVCG_CACHE_CUH

class Cache {
public:
    typedef kvgpu::KVCacheWrapper<1000000, 8> type;
};

#endif //KVCG_CACHE_CUH
