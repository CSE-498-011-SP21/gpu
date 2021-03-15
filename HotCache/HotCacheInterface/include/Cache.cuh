/**
 * @file
 */

#ifndef KVCG_CACHE_CUH
#define KVCG_CACHE_CUH

/**
 * Type to be used by GPU.
 */
class Cache {
public:
    typedef kvgpu::KVCacheWrapper<1000000, 8> type;
};

#endif //KVCG_CACHE_CUH
