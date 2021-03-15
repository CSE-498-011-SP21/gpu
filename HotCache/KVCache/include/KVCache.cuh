//
// Created by depaulsmiller on 8/28/20.
//

#ifndef KVGPU_KVCACHE_CUH
#define KVGPU_KVCACHE_CUH

#include <mutex>
#include <atomic>
#include <functional>
#include <iostream>
#include <shared_mutex>
#include <data_t.hh>
#include <tbb/concurrent_vector.h>
#include <immintrin.h>
#include <Model.hh>
#include <ImportantDefinitions.hh>
#include <RequestTypes.hh>

namespace kvgpu {

    using mutex = std::shared_mutex;
    typedef std::unique_lock<mutex> locktype;
    typedef std::shared_lock<mutex> sharedlocktype;

    template<typename K, typename V>
    struct LockingPair {
        LockingPair() : valid(0), deleted(0), key(), value() {}

        ~LockingPair() {}

        char padding[40];
        unsigned long valid;
        unsigned long deleted;
        K key;
        V value;
    };

    /**
     * KVCache caches keys and values
     * K is the key type
     * V is the value type
     * DSCaching is a data structure that is being cached by this cache
     * SETS is the number of SETs in the cache
     * N is the number of elements per set
     * @tparam K
     * @tparam V
     * @tparam DSCaching
     * @tparam SETS
     * @tparam N
     */
    template<typename K, typename V, unsigned SETS = 524288 / sizeof(LockingPair<K, V>) / 8, unsigned N = 8>
    class KVCache {
    private:
        struct Node_t {
            explicit Node_t(int startLoc) : loc(startLoc), set(new LockingPair<K, V>[N]), next(nullptr) {

                for (int j = 0; j < N; j++) {
                    set[j].valid = 0;
                }
            }

            ~Node_t() {
                delete[] set;
            }

            int loc;
            LockingPair<K, V> *set;
            std::atomic<Node_t *> next;
        };

    public:
        /**
         * Creates cache
         */
        KVCache() : log_requests(new tbb::concurrent_vector<int>(N * SETS)),
                    log_hash(new tbb::concurrent_vector<unsigned>(N * SETS)),
                    log_keys(new tbb::concurrent_vector<K>(N * SETS)),
                    log_values(new tbb::concurrent_vector<V>(N * SETS)),
                    log_size(N * SETS),
                    map(new LockingPair<K, V> *[SETS]),
                    mtx(new mutex[SETS]),
                    nodes(new std::atomic<Node_t *>[SETS]),
                    expansions(0) {
            for (int i = 0; i < SETS; i++) {
                nodes[i] = nullptr;
                map[i] = new LockingPair<K, V>[N];
                for (int j = 0; j < N; j++) {
                    std::unique_lock<mutex> ul(mtx[i]);
                    map[i][j].valid = 0;
                    map[i][j].value = 0;
                }
            }
        }

        /**
         * Removes cache
         */
        ~KVCache() {
            for (int i = 0; i < SETS; i++) {
                delete[] map[i];
            }
            delete[] map;
            delete[] nodes;
            delete[] mtx;
            delete log_requests;
            delete log_hash;
            delete log_keys;
            delete log_values;
        }


        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::pair<LockingPair<K, V> *, sharedlocktype> fast_get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            sharedlocktype sharedlock(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    return {&set[i], std::move(sharedlock)};
                }
            }
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        return {&set[i], std::move(sharedlock)};
                    }
                }
                node = node->next;
            }

            return {firstInvalidPair, std::move(std::shared_lock<std::shared_mutex>())};
        }


        /**
         *
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *put(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            size_t logLoc;
            auto pair = get_with_log(key, hash, mfn, logLoc);

            data_t *old_value = pair.first->value;

            pair.first->value = value;
            pair.first->deleted = 0;
            pair.first->valid = 1;
            log_requests->operator[](logLoc) = REQUEST_INSERT;
            log_hash->operator[](logLoc) = hash;
            log_keys->operator[](logLoc) = key;
            log_values->operator[](logLoc) = value;

            return old_value;
        }

        /**
         *
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *remove(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            size_t logLoc;
            auto pair = get_with_log(key, hash, mfn, logLoc);

            data_t *old_value = pair.first->value;
            pair.first->value = nullptr;
            pair.first->deleted = 1;
            pair.first->valid = 1;

            log_requests->operator[](logLoc) = REQUEST_REMOVE;
            log_hash->operator[](logLoc) = hash;
            log_keys->operator[](logLoc) = key;

            return old_value;
        }


        /**
         * Returns value that should be written to client. Insert (do not do PUT) the value into the map.
         * Return a copy of what is in the map at the end of the insert.
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *missCallback(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            auto cacheRes = internal_get(key, hash, mfn);

            data_t *cpy = nullptr;

            if (cacheRes.first->valid == 1) {
                if (cacheRes.first->deleted == 0) {
                    cpy = new data_t(cacheRes.first->value->size);
                    memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                }
            } else {
                cacheRes.first->valid = 1;
                cacheRes.first->value = value;
                cacheRes.first->deleted = (value == EMPTY<V>::value);
                if (cacheRes.first->deleted == 0) {
                    cpy = new data_t(cacheRes.first->value->size);
                    memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                }
            }

            return cpy;
        }

        template<typename H>
        void scan_and_evict(const Model<K> &mfn, const H &hfn, std::unique_lock<std::mutex> modelLock) {

            for (int setIdx = 0; setIdx < SETS; setIdx++) {
                LockingPair<K, V> *set = map[setIdx];
                locktype unique(mtx[setIdx]);

                for (unsigned i = 0; i < N; i++) {
                    if (!mfn(set[i].key, hfn(set[i].key))) {
                        set[i].valid = 0;
                    }
                }
                Node_t *node = nodes[setIdx].load();
                while (node != nullptr) {
                    set = node->set;
                    for (unsigned i = 0; i < N; i++) {

                        if (!mfn(set[i].key, hfn(set[i].key))) {
                            set[i].valid = 0;
                        }
                    }
                    node = node->next;
                }
            }
        }


        constexpr size_t getN() {
            return N;
        }

        constexpr size_t getSETS() {
            return SETS;
        }

        void stat() {
            //std::cout << "Cache Expansions " << expansions << std::endl;
            //std::cout << "Footprint without expansions: " << (sizeof(*this) + sizeof(LockingPair<K,V>) * SETS * N) / 1024.0 / 1024.0 << " MB" << std::endl;
        }

        tbb::concurrent_vector<int> *log_requests;
        tbb::concurrent_vector<unsigned> *log_hash;
        tbb::concurrent_vector<K> *log_keys;
        tbb::concurrent_vector<V> *log_values;
        std::atomic_size_t log_size;


        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::pair<LockingPair<K, V> *, locktype> internal_get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    return {&set[i], std::move(unique)};
                } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                    set[i].valid = 0;
                    firstInvalidPair = &set[i];
                }
            }
            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        return {&set[i], std::move(unique)};
                    } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                        set[i].valid = 0;
                        firstInvalidPair = &set[i];
                    }
                }
                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(N);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                if (prevNode != nullptr) {
                    prevNode->next = new Node_t(tmploc);
                    node = prevNode->next;
                } else {
                    nodes[setIdx].store(new Node_t(tmploc));
                    node = nodes[setIdx].load();
                }
                expansions++;
                firstInvalidPair = &(node->set[0]);
            }

            firstInvalidPair->valid = 2;
            firstInvalidPair->key = key;

            return {firstInvalidPair, std::move(unique)};
        }


        std::pair<LockingPair<K, V> *, locktype>
        get_with_log(K key, unsigned hash, const Model<K> &mfn, size_t &logLoc) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    logLoc = setIdx * N + i;
                    return {&set[i], std::move(unique)};
                } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                    set[i].valid = 0;
                    firstInvalidPair = &set[i];
                }
            }
            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        logLoc = setIdx * N + i;
                        return {&set[i], std::move(unique)};
                    } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                        set[i].valid = 0;
                        firstInvalidPair = &set[i];
                    }
                }
                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(N);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                if (prevNode != nullptr) {
                    prevNode->next = new Node_t(tmploc);
                    node = prevNode->next;
                } else {
                    nodes[setIdx].store(new Node_t(tmploc));
                    node = nodes[setIdx].load();
                }
                expansions++;
                firstInvalidPair = &(node->set[0]);
                logLoc = tmploc;
            }

            firstInvalidPair->valid = 2;
            firstInvalidPair->key = key;

            return {firstInvalidPair, std::move(unique)};
        }

    private:

        LockingPair<K, V> **map;
        mutex *mtx;
        std::atomic<Node_t *> *nodes;
        std::atomic_size_t expansions;
    };

    /**
     * KVCache caches keys and values
     * K is the key type
     * V is the value type
     * DSCaching is a data structure that is being cached by this cache
     * SETS is the number of SETs in the cache
     * N is the number of elements per set
     * @tparam K
     * @tparam V
     * @tparam DSCaching
     * @tparam SETS
     * @tparam N
     */
    template<unsigned SETS = 524288 / sizeof(LockingPair<unsigned long long, data_t *>) / 8, unsigned N = 8>
    class KVCacheWrapper {
    private:
        using K = unsigned long long;
    public:
        /**
         * Creates cache
         */
        KVCacheWrapper()
                : cache(), log_requests(cache.log_requests), log_hash(cache.log_hash), log_keys(cache.log_keys),
                  log_values(cache.log_values), log_size(cache.log_size) {

        }

        /**
         * Removes cache
         */
        ~KVCacheWrapper() {}

        std::pair<bool, data_t *>get(K key, unsigned hash, const Model<K> &mfn) {
            auto pair = cache.fast_get(key, hash, mfn);

            if (pair.first == nullptr || pair.first->valid != 1) {
                return {true, nullptr};
            } else {
                auto cpy = new data_t(pair.first->value->size);
                memcpy(cpy->data, pair.first->value->data, cpy->size);
                return {false, cpy};
            }
        }

        /**
         *
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *put(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            size_t logLoc;
            auto pair = cache.get_with_log(key, hash, mfn, logLoc);

            data_t *old_value = pair.first->value;

            pair.first->value = value;
            pair.first->deleted = 0;
            pair.first->valid = 1;
            log_requests->operator[](logLoc) = REQUEST_INSERT;
            log_hash->operator[](logLoc) = hash;
            log_keys->operator[](logLoc) = key;
            log_values->operator[](logLoc) = value;

            return old_value;
        }

        /**
         *
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *remove(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            size_t logLoc;
            auto pair = cache.get_with_log(key, hash, mfn, logLoc);

            data_t *old_value = pair.first->value;
            pair.first->value = nullptr;
            pair.first->deleted = 1;
            pair.first->valid = 1;

            log_requests->operator[](logLoc) = REQUEST_REMOVE;
            log_hash->operator[](logLoc) = hash;
            log_keys->operator[](logLoc) = key;

            return old_value;
        }


        /**
         * Returns value that should be written to client. Insert (do not do PUT) the value into the map.
         * Return a copy of what is in the map at the end of the insert.
         * @param key
         * @param value
         * @param hash
         * @param mfn
         * @return
         */
        inline data_t *missCallback(K key, data_t *value, unsigned hash, const Model<K> &mfn) {
            auto cacheRes = cache.internal_get(key, hash, mfn);

            data_t *cpy = nullptr;

            if (cacheRes.first->valid == 1) {
                if (cacheRes.first->deleted == 0) {
                    cpy = new data_t(cacheRes.first->value->size);
                    memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                }
            } else {
                cacheRes.first->valid = 1;
                cacheRes.first->value = value;
                cacheRes.first->deleted = (value == nullptr);
                if (cacheRes.first->deleted == 0) {
                    cpy = new data_t(cacheRes.first->value->size);
                    memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                }
            }

            return cpy;
        }

        template<typename H>
        void scan_and_evict(const Model<K> &mfn, const H &hfn, std::unique_lock<std::mutex> modelLock) {
            cache.scan_and_evict(mfn, hfn, std::move(modelLock));
        }


        constexpr size_t getN() {
            return N;
        }

        constexpr size_t getSETS() {
            return SETS;
        }

        void stat() {
            //std::cout << "Cache Expansions " << expansions << std::endl;
            //std::cout << "Footprint without expansions: " << (sizeof(*this) + sizeof(LockingPair<K,V>) * SETS * N) / 1024.0 / 1024.0 << " MB" << std::endl;
        }

    private:

        KVCache<K, data_t *, SETS, N> cache;

    public:

        tbb::concurrent_vector<int> *&log_requests;
        tbb::concurrent_vector<unsigned> *&log_hash;
        tbb::concurrent_vector<K> *&log_keys;
        tbb::concurrent_vector<data_t *> *&log_values;
        std::atomic_size_t &log_size;
    };


    /**
     * KVCache caches keys and values
     * K is the key type
     * V is the value type
     * DSCaching is a data structure that is being cached by this cache
     * SETS is the number of SETs in the cache
     * N is the number of elements per set
     * @tparam K
     * @tparam V
     * @tparam DSCaching
     * @tparam SETS
     * @tparam N
     */
    template<typename K, typename V, unsigned SETS = 524288 / sizeof(LockingPair<K, V>) / 8>
    class KVSimdCache {
        static_assert(sizeof(K) == sizeof(uint64_t), "SIMD cache needs 8B entries");
    private:

        struct Bucket {
            Bucket() {
                uint8_t *v = &valid;
                *v = 0;
                uint8_t *d = &deleted;
                *d = 0;
            }

            ~Bucket() {}

            K key[8];
            struct {
                unsigned bit: 1;
            } valid[8];

            struct {
                unsigned bit: 1;
            } deleted[8];

            V value[8];
        };


        struct Node_t {
            explicit Node_t(int startLoc) : loc(startLoc), set(), next(nullptr) {
            }

            ~Node_t() {
                delete[] set;
            }

            Bucket set;
            int loc;
            std::atomic<Node_t *> next;
        };

    public:
        /**
         * Creates cache
         */
        KVSimdCache() : log_requests(new tbb::concurrent_vector<int>(8 * SETS)),
                        log_hash(new tbb::concurrent_vector<unsigned>(8 * SETS)),
                        log_keys(new tbb::concurrent_vector<K>(8 * SETS)),
                        log_values(new tbb::concurrent_vector<V>(8 * SETS)),
                        log_size(8 * SETS),
                        map(new Bucket *[SETS]),
                        mtx(new mutex[SETS]),
                        nodes(new std::atomic<Node_t *>[SETS]),
                        expansions(0) {
            for (int i = 0; i < SETS; i++) {
                nodes[i] = nullptr;
                map[i] = new LockingPair<K, V>[8];
                for (int j = 0; j < 8; j++) {
                    std::unique_lock<mutex> ul(mtx[i]);
                    map[i][j].valid = 0;
                    map[i][j].value = 0;
                }
            }
        }

        /**
         * Removes cache
         */
        ~KVSimdCache() {
            for (int i = 0; i < SETS; i++) {
                delete[] map[i];
            }
            delete[] map;
            delete[] nodes;
            delete[] mtx;
            delete log_requests;
            delete log_hash;
            delete log_keys;
            delete log_values;
        }


        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::tuple<Bucket *, int, locktype> get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            Bucket *firstInvalidPair = nullptr;
            int firstInvalidPairIdx = 0;

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        return {set, i + 4 * k, std::move(unique)};
                    } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                        firstInvalidPair = set;
                        firstInvalidPairIdx = i + 4 * k;
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            return {set, i + 4 * k, std::move(unique)};
                        } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                            firstInvalidPair = set;
                            firstInvalidPairIdx = i + 4 * k;
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(8);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                node = prevNode->next;
                firstInvalidPair = &(node->set);
                firstInvalidPairIdx = 0;
            }

            firstInvalidPair->valid[firstInvalidPairIdx] = 2;
            firstInvalidPair->key[firstInvalidPairIdx] = key;

            return {firstInvalidPair, firstInvalidPairIdx, std::move(unique)};
        }

        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::tuple<Bucket *, int, locktype> fast_get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            sharedlocktype unique(mtx[setIdx]);

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        return {set, i + 4 * k, std::move(unique)};
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            return {set, i + 4 * k, std::move(unique)};
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            return {nullptr, 0, std::move(sharedlocktype())};
        }


        std::tuple<Bucket *, int, locktype>
        get_with_log(K key, unsigned hash, const Model<K> &mfn, size_t &logLoc) {

            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            Bucket *firstInvalidPair = nullptr;
            int firstInvalidPairIdx = 0;

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        logLoc = setIdx * 8 + (i + 4 * k);
                        return {set, i + 4 * k, std::move(unique)};
                    } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                        firstInvalidPair = set;
                        firstInvalidPairIdx = i + 4 * k;
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            logLoc = setIdx * 8 + (i + 4 * k);
                            return {set, i + 4 * k, std::move(unique)};
                        } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                            firstInvalidPair = set;
                            firstInvalidPairIdx = i + 4 * k;
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(8);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                logLoc = tmploc;
                node = prevNode->next;
                firstInvalidPair = &(node->set);
                firstInvalidPairIdx = 0;
            }

            firstInvalidPair->valid[firstInvalidPairIdx] = 2;
            firstInvalidPair->key[firstInvalidPairIdx] = key;

            return {firstInvalidPair, firstInvalidPairIdx, std::move(unique)};
        }

        template<typename H>
        void scan_and_evict(const Model<K> &mfn, const H &hfn, std::unique_lock<std::mutex> modelLock) {

            for (int setIdx = 0; setIdx < SETS; setIdx++) {
                LockingPair<K, V> *set = map[setIdx];
                locktype unique(mtx[setIdx]);

                for (unsigned i = 0; i < 8; i++) {
                    if (!mfn(set[i].key, hfn(set[i].key))) {
                        set[i].valid = 0;
                    }
                }
                Node_t *node = nodes[setIdx].load();
                while (node != nullptr) {
                    set = node->set;
                    for (unsigned i = 0; i < 8; i++) {

                        if (!mfn(set[i].key, hfn(set[i].key))) {
                            set[i].valid = 0;
                        }
                    }
                    node = node->next;
                }
            }
        }


        constexpr size_t getN() {
            return 8;
        }

        constexpr size_t getSETS() {
            return SETS;
        }

        void stat() {
            //std::cout << "Cache Expansions " << expansions << std::endl;
            //std::cout << "Footprint without expansions: " << (sizeof(*this) + sizeof(LockingPair<K,V>) * SETS * N) / 1024.0 / 1024.0 << " MB" << std::endl;
        }

        tbb::concurrent_vector<int> *log_requests;
        tbb::concurrent_vector<unsigned> *log_hash;
        tbb::concurrent_vector<K> *log_keys;
        tbb::concurrent_vector<V> *log_values;
        std::atomic_size_t log_size;

    private:
        Bucket **map;
        mutex *mtx;
        std::atomic<Node_t *> *nodes;
        std::atomic_size_t expansions;
    };

}

#endif //KVGPU_KVCACHE_CUH
