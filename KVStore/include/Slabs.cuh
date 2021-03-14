//
// Created by depaulsmiller on 3/14/21.
//

#include <gpuSystemConfig.cuh>
#include <BatchData.hh>
#include "StatData.cuh"
#include "Cache.cuh"

#ifndef KVCG_SLABS_HH
#define KVCG_SLABS_HH

const int MAX_ATTEMPTS = 1;

template<typename K, typename V, typename M>
struct Slabs {

    using VType = V;

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData < K, V> *>
    q_t;

    Slabs(const std::vector <PartitionedSlabUnifiedConfig> &config, std::shared_ptr<typename Cache<K, V>::type> cache,
          std::shared_ptr <M> m) : numslabs(config.size()), slabs(new SlabUnified<K, V>[numslabs]),
                                   gpu_qs(new q_t[numslabs]), done(false),
                                   mops(new tbb::concurrent_vector<StatData>[numslabs]), _cache(cache), ops(0), load(0),
                                   model(m) {
        for (int i = 0; i < config.size(); i++) {
            cudaStream_t *stream = new cudaStream_t();
            *stream = config[i].stream;
            slabs[i] = std::move(SlabUnified<K, V>(config[i].size, config[i].gpu, stream));
        }
        for (int i = 0; i < numslabs; ++i) {
            threads.push_back(std::thread([this](int tid) {
                K *keys = slabs[tid].getBatchKeys();
                V *values = slabs[tid].getBatchValues();
                int *requests = slabs[tid].getBatchRequests();
                unsigned *hashes = slabs[tid].getHashValues();

                BatchData <K, V> *holdonto = nullptr;

                std::vector < std::pair < int, BatchData < K, V > * >> writeBack;
                writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                int index = THREADS_PER_BLOCK * BLOCKS;
                while (!done.load()) {
                    writeBack.clear();
                    for (int i = 0; i < index; i++) {
                        requests[i] = REQUEST_EMPTY;
                    }
                    index = 0;

                    BatchData <K, V> *res;

                    auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                    if (holdonto) {
                        //std::cerr << "Hold onto set " << tid << std::endl;
                        writeBack.push_back({index, holdonto});

                        for (int i = 0; i < holdonto->idx; i++) {
                            keys[index + i] = holdonto->keys[i];
                            values[index + i] = holdonto->values[i];
                            requests[index + i] = holdonto->requests[i];
                            hashes[index + i] = holdonto->hashes[i];
                        }
                        index += holdonto->idx;
                        holdonto = nullptr;
                    }

                    int attempts = 0;

                    while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                        if (this->gpu_qs[tid].try_pop(res)) {
                            load--;
                            //std::cerr << "Got a batch on handler thread " << tid << "\n";
                            if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                holdonto = res;
                                break;
                            }
                            for (int i = 0; i < res->idx; i++) {
                                keys[index + i] = res->keys[i];
                                values[index + i] = res->values[i];
                                requests[index + i] = res->requests[i];
                                hashes[index + i] = res->hashes[i];
                            }
                            writeBack.push_back({index, res});
                            index += res->idx;
                            if (res->flush) {
                                break;
                            }
                        } else {
                            attempts++;
                        }
                    }

                    if (index > 0) {

                        //std::cerr << "Batching " << tid << "\n";

                        auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                        float t;

                        this->slabs[tid].diy_batch(t, ceil(index / 512.0), 512);

                        auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                        int timesGoingToCache = 0;
                        for (auto &wb : writeBack) {

                            int rbLoc = wb.second->resBufStart;

                            for (int i = 0; i < wb.second->idx; ++i) {

                                if (wb.second->handleInCache[i]) {
                                    timesGoingToCache++;
                                    auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                *(this->model));
                                    if (cacheRes.first->valid == 1) {
                                        wb.second->resBuf->resultValues[rbLoc + i] = cacheRes.first->value;
                                    } else {
                                        cacheRes.first->valid = 1;
                                        cacheRes.first->value = values[wb.first + i];
                                        cacheRes.first->deleted = (values[wb.first + i] == EMPTY<V>::value);
                                    }
                                    asm volatile("":: : "memory");

                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                } else {
                                    wb.second->resBuf->resultValues[rbLoc + i] = values[wb.first + i];
                                    asm volatile("":: : "memory");
                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                }
                            }
                            delete wb.second;
                        }

                        mops[tid].push_back(
                                {std::chrono::high_resolution_clock::now(), std::chrono::high_resolution_clock::now(),
                                 timestampWriteBack, timestampStartBatch,
                                 timestampWriteToBatch, t, index, timesGoingToCache});

                        ops += index;
                        //std::cerr << "Batched " << tid << "\n";

                    }
                }
            }, i));
        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] slabs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    SlabUnified <K, V> *slabs;
    q_t *gpu_qs;
    std::vector <std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector <StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr <M> model;
};

template<typename K, typename M>
struct Slabs<K, data_t *, M> {

    using V = data_t *;
    using VType = data_t *;

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData < K, data_t> *>
    q_t;

    Slabs(const std::vector <PartitionedSlabUnifiedConfig> &config,
          std::shared_ptr<typename Cache<K, V>::type> cache, std::shared_ptr <M> m) : done(false),
                                                                                      mops(new tbb::concurrent_vector<StatData>[config.size()]),
                                                                                      _cache(cache), ops(0),
                                                                                      load(0), model(m) {
        std::unordered_map < int, std::shared_ptr < SlabUnified < K, V>>> gpusToSlab;
        for (int i = 0; i < config.size(); i++) {
            if (gpusToSlab.find(config[i].gpu) == gpusToSlab.end())
                gpusToSlab[config[i].gpu] = std::make_shared < SlabUnified < K, V >> (config[i].size, config[i].gpu);
        }
        gpu_qs = new q_t[gpusToSlab.size()];
        numslabs = gpusToSlab.size();

        for (int i = 0; i < config.size(); i++) {
            //config[i].stream;
            threads.push_back(
                    std::thread([this](int tid, int gpu, std::shared_ptr <SlabUnified<K, V>> slab,
                                       cudaStream_t stream) {
                                    slab->setGPU();
                                    auto batchData = new BatchBuffer<K, V>();

                                    K *keys = batchData->getBatchKeys();
                                    V *values = batchData->getBatchValues();
                                    int *requests = batchData->getBatchRequests();
                                    unsigned *hashes = batchData->getHashValues();

                                    BatchData <K, data_t> *holdonto = nullptr;

                                    std::vector < std::pair < int, BatchData < K, data_t > * >> writeBack;
                                    writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);
                                    std::chrono::high_resolution_clock::time_point sampleTime;
                                    bool sampleTimeSet = false;
                                    int index = THREADS_PER_BLOCK * BLOCKS;
                                    while (!done.load()) {
                                        writeBack.clear();
                                        for (int i = 0; i < index; i++) {
                                            requests[i] = REQUEST_EMPTY;
                                        }
                                        index = 0;

                                        BatchData <K, data_t> *res;

                                        auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                                        if (holdonto) {
                                            //std::cerr << "Hold onto set " << tid << std::endl;
                                            writeBack.push_back({index, holdonto});

                                            for (int i = 0; i < holdonto->idx; i++) {
                                                keys[index + i] = holdonto->keys[i];
                                                values[index + i] = holdonto->values[i];
                                                requests[index + i] = holdonto->requests[i];
                                                hashes[index + i] = holdonto->hashes[i];
                                            }
                                            index += holdonto->idx;
                                            holdonto = nullptr;
                                            sampleTime = holdonto->start;
                                            sampleTimeSet = true;
                                        }

                                        int attempts = 0;

                                        while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                                            if (this->gpu_qs[gpu].try_pop(res)) {
                                                load--;
                                                //std::cerr << "Got a batch on handler thread " << tid << "\n";
                                                if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                                    //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                                    holdonto = res;
                                                    break;
                                                }
                                                for (int i = 0; i < res->idx; i++) {
                                                    keys[index + i] = res->keys[i];
                                                    values[index + i] = res->values[i];
                                                    assert(res->requests[i] != REQUEST_INSERT || res->values[i]->size > 0);
                                                    requests[index + i] = res->requests[i];
                                                    hashes[index + i] = res->hashes[i];
                                                }
                                                writeBack.push_back({index, res});
                                                index += res->idx;
                                                if (!sampleTimeSet) {
                                                    sampleTime = res->start;
                                                    sampleTimeSet = true;
                                                }

                                                if (res->flush) {
                                                    break;
                                                }
                                            } else {
                                                attempts++;
                                            }
                                        }
                                        sampleTimeSet = false;

                                        if (index > 0) {

                                            //std::cerr << "Batching " << tid << "\n";

                                            auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                                            cudaEvent_t start, stop;

                                            gpuErrchk(cudaEventCreate(&start));
                                            gpuErrchk(cudaEventCreate(&stop));

                                            float t;

                                            slab->moveBufferToGPU(batchData, stream);
                                            gpuErrchk(cudaEventRecord(start, stream));
                                            slab->diy_batch(batchData, ceil(index / 512.0), 512, stream);
                                            gpuErrchk(cudaEventRecord(stop, stream));
                                            slab->moveBufferToCPU(batchData, stream);
                                            gpuErrchk(cudaStreamSynchronize(stream));

                                            auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                                            gpuErrchk(cudaEventElapsedTime(&t, start, stop));
                                            gpuErrchk(cudaEventDestroy(start));
                                            gpuErrchk(cudaEventDestroy(stop));
                                            int timesGoingToCache = 0;
                                            for (auto &wb : writeBack) {

                                                int rbLoc = wb.second->resBufStart;

                                                for (int i = 0; i < wb.second->idx; ++i) {

                                                    if (wb.second->handleInCache[i]) {
                                                        timesGoingToCache++;
                                                        auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                                    *(this->model));
                                                        if (cacheRes.first->valid == 1) {
                                                            V cpy = nullptr;
                                                            if (cacheRes.first->deleted == 0) {
                                                                cpy = new data_t(cacheRes.first->value->size);
                                                                memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                                                            }

                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            cacheRes.first->valid = 1;
                                                            cacheRes.first->value = values[wb.first + i];
                                                            cacheRes.first->deleted = (values[wb.first + i] ==
                                                                                       EMPTY<V>::value);
                                                        }
                                                        asm volatile("":: : "memory");

                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                                    } else {
                                                        if (requests[wb.first + i] == REQUEST_REMOVE) {
                                                            wb.second->resBuf->resultValues[rbLoc +
                                                                                            i] = values[wb.first + i];
                                                        } else if (requests[wb.first + i] == REQUEST_GET) {
                                                            V cpy = nullptr;
                                                            if (values[wb.first + i]) {
                                                                cpy = new data_t(values[wb.first + i]->size);
                                                                memcpy(cpy->data, values[wb.first + i]->data, cpy->size);
                                                            }
                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            wb.second->resBuf->resultValues[rbLoc + i] = nullptr;
                                                        }

                                                        asm volatile("":: : "memory");
                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                                    }
                                                }
                                                delete wb.second;
                                            }

                                            mops[gpu].push_back({sampleTime,
                                                                        /*timestampEnd = */
                                                                 std::chrono::high_resolution_clock::now(),
                                                                        /*timestampWriteBack = */ timestampWriteBack,
                                                                        /*timestampStartBatch = */timestampStartBatch,
                                                                        /*timestampDequeueToBatch = */timestampWriteToBatch,
                                                                        /*duration*/t,
                                                                        /*size = */index,
                                                                        /*timesGoingToCache=*/ timesGoingToCache});

                                            ops += index;
                                            //std::cerr << "Batched " << tid << "\n";

                                        }
                                    }
                                    if (stream != cudaStreamDefault) gpuErrchk(cudaStreamDestroy(stream));
                                }, i, config[i].gpu,
                                gpusToSlab[config[i].gpu], config[i].stream));

        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    q_t *gpu_qs;
    std::vector <std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector <StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr <M> model;
};

#endif //KVCG_SLABS_HH
