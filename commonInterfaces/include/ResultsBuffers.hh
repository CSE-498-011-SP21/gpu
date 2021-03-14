//
// Created by depaulsmiller on 3/14/21.
//

#include "data_t.hh"

#ifndef KVCG_RESULTSBUFFERS_HH
#define KVCG_RESULTSBUFFERS_HH

template<typename V>
struct ResultsBuffers {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new V[s]), size(s), retryGPU(false),
                                     retryModifications(false) {
        for (int i = 0; i < size; i++)
            requestIDs[i] = -1;
    }

    ResultsBuffers(const ResultsBuffers<V> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile V *resultValues;
    int size;
    bool retryGPU;
    bool retryModifications;
};

template<>
struct ResultsBuffers<data_t> {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new volatile data_t *[s]), size(s),
                                     retryGPU(false), retryModifications(false) {
        for (int i = 0; i < size; i++) {
            requestIDs[i] = -1;
            resultValues[i] = nullptr;
        }
    }

    ResultsBuffers(const ResultsBuffers<data_t> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        //for (int i = 0; i < size; i++) {
        //    if (resultValues[i] && resultValues[i]->data)
        //delete[] resultValues[i]->data;
        //}
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile data_t **resultValues;
    int size;
    bool retryGPU;
    bool retryModifications;
};

#endif //KVCG_RESULTSBUFFERS_HH
