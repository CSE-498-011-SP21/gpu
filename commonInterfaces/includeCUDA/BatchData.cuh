/**
 * @file
 */

#include <Communication.hh>

#ifndef KVCG_BATCHDATA_HH
#define KVCG_BATCHDATA_HH

template<typename K>
struct BatchData {
    BatchData(int rbStart, std::shared_ptr<Communication> &rb, int s,
              std::chrono::high_resolution_clock::time_point start_) : keys(s), values(s), requests(s),
                                                                       hashes(s), requestID(s),
                                                                       handleInCache(s), resBuf(rb),
                                                                       resBufStart(rbStart), size(s), idx(0),
                                                                       flush(false), start(start_) {
        for (int i = 0; i < s; i++) {
            handleInCache[i] = false;
        }
    }

    ~BatchData() = default;

    std::vector<K> keys;
    std::vector<data_t *> values;
    std::vector<unsigned> requests;
    std::vector<unsigned> hashes;
    std::vector<int> requestID;
    std::vector<bool> handleInCache;
    std::shared_ptr<Communication> resBuf;
    int resBufStart;
    int size;
    int idx;
    bool flush;
    std::chrono::high_resolution_clock::time_point start;
};

#endif //KVCG_BATCHDATA_HH
