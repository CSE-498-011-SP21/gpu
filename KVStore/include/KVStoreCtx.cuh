//
// Created by depaulsmiller on 9/2/20.
//

#include "KVStoreInternalClient.cuh"

#ifndef KVGPU_KVSTORECTX_CUH
#define KVGPU_KVSTORECTX_CUH

template<typename K, typename V, typename M>
class KVStoreCtx {
private:
    using KVStore_t = KVStore<K, typename SubstituteType<V>::type, M>;
public:

    using Slab_t = typename KVStore_t::Slab_t;

    KVStoreCtx() : k() {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig> &conf, int num_threads) : k(conf), num_threads_(num_threads) {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig> &conf, int num_threads, const std::string& modelFile) : k(conf, modelFile), num_threads_(num_threads) {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig> &conf, int num_threads, M& m) : k(conf, m), num_threads_(num_threads) {

    }


    ~KVStoreCtx() {}

    std::unique_ptr<KVStoreInternalClient<K, V, M, Slab_t>> getClient() {
        return std::make_unique<KVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel(), num_threads_);
    }

    std::unique_ptr<NoCacheKVStoreInternalClient<K, V, M, Slab_t>> getNoCacheClient() {
        return std::make_unique<NoCacheKVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel(), num_threads_);
    }

    std::unique_ptr<JustCacheKVStoreInternalClient<K, V, M, Slab_t>> getJustCacheClient() {
        return std::make_unique<JustCacheKVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel(), num_threads_);
    }

private:
    KVStore_t k;
    int num_threads_;
};

#endif //KVGPU_KVSTORECTX_CUH
