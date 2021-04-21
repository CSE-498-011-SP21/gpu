/**
 * @file
 */

#include <gpuSystemConfig.cuh>
#include <BatchData.cuh>
#include "StatData.cuh"
#include <Cache.hh>
#include "GpuBTree.h"

#ifndef KVCG_BTREES_HH
#define KVCG_BTREES_HH

/**
 * Copy of Slabs. This is to serve as a BTree container.
 * @tparam M the Model
 */
template<typename M>
struct BTrees {
    const int MAX_ATTEMPTS = 1;

    using V = data_t *;
    using VType = data_t *;

    BTrees() = delete;

    typedef tbb::concurrent_queue<BatchData<unsigned long long> *> q_t;

    /**
     * This is a pool of threads which read from a
     * @param config Vector of configs, one per CUDA Stream (per GPU as well?)
     * @param cache KVCacheWrapper defined in Cache.hh, The Hot Cache on the CPU?
     * @param m The Model, AnalyticalModel in Model.hh
     */
    BTrees(const std::vector<PartitionedSlabUnifiedConfig> &config,
          std::shared_ptr<typename Cache::type> cache, std::shared_ptr<M> m) : done(false),
                                                                                  mops(new tbb::concurrent_vector<StatData>[config.size()]),
                                                                                  _cache(cache), ops(0),
                                                                                  load(0), model(m) {
        std::unordered_map<int, std::shared_ptr<SlabUnified<unsigned long long, V>>> gpusToSlab;

        // TODO: Find a clean way to enforce the single stream requirement statically.
        assert(("BTree is only correct with a single CUDA Stream.", config.size() == 1));

        // Populates the map. If a SlabUnified is not assigned to the GPU, create one and assign it.
        for (auto i : config) {
            if (gpusToSlab.find(i.gpu) == gpusToSlab.end())
                gpusToSlab[i.gpu] = std::make_shared<SlabUnified<unsigned long long, V >>(i.size, i.gpu);
        }
        // This is the queue that this.batch() adds to.
        gpu_qs = new q_t[gpusToSlab.size()];
        numslabs = gpusToSlab.size();

        // For each stream,
        for (int i = 0; i < config.size(); i++) {
            //config[i].stream;
            threads.push_back(
                    std::thread([this](int tid, int gpu, std::shared_ptr<SlabUnified<unsigned long long, V>> slab,
                                       cudaStream_t stream) {
                                    slab->setGPU();
                                    auto batchData = new BatchBuffer<unsigned long long, V>();

                                    unsigned long long *keys = batchData->getBatchKeys();
                                    V *values = batchData->getBatchValues();
                                    int *requests = batchData->getBatchRequests();
                                    unsigned *hashes = batchData->getHashValues();

                                    BatchData<unsigned long long> *holdonto = nullptr;

                                    std::vector<std::pair<int, BatchData<unsigned long long> * >> writeBack;
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

                                        BatchData<unsigned long long> *res;

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
                                            sampleTime = holdonto->start;
                                            holdonto = nullptr;
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

                                                for (int i = 0; i < wb.second->idx; ++i) {

                                                    if (wb.second->handleInCache[i]) {
                                                        timesGoingToCache++;

                                                        auto value = _cache->missCallback(
                                                                wb.second->keys[i], values[wb.first + i], wb.second->hashes[i],
                                                                *(this->model));
                                                        wb.second->resBuf->send(Response(wb.second->requestID[i], value, false));

                                                    } else {
                                                        data_t* value;
                                                        if (requests[wb.first + i] == REQUEST_REMOVE) {
                                                            value = values[wb.first + i];

                                                        } else if (requests[wb.first + i] == REQUEST_GET) {
                                                            V cpy = nullptr;
                                                            if (values[wb.first + i]) {
                                                                cpy = new data_t(values[wb.first + i]->size);
                                                                memcpy(cpy->data, values[wb.first + i]->data, cpy->size);
                                                            }
                                                            value = cpy;
                                                        } else {
                                                            value = nullptr;
                                                        }
                                                        wb.second->resBuf->send(Response(wb.second->requestID[i], value, false));
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

    ~BTrees() {
        std::cerr << "BTrees deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] mops;
    }

    inline void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    inline size_t getOps() {
        return ops;
    }

    inline void batch(BatchData<unsigned long long> *b, int partition) {
        gpu_qs[partition].push(b);
    }

    inline void increaseLoad() {
        load++;
    }

    inline size_t getLoad() {
        return load.load();
    }

    inline tbb::concurrent_vector<StatData> *getMops() {
        return mops;
    }

    inline int getNumslabs() {
        return numslabs;
    }

private:
    std::atomic_int load;
    /// Work queue for the threads to take from
    q_t *gpu_qs;
    int numslabs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    /// Holds timing data
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache::type> _cache;
    std::atomic_size_t ops;
    std::shared_ptr<M> model;
};

#endif //KVCG_BTREES_HH
