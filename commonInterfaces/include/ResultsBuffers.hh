/**
 * @file
 */

#include "data_t.hh"
#include "Messages.hh"
#include <tbb/concurrent_queue.h>

#ifndef KVCG_RESULTSBUFFERS_HH
#define KVCG_RESULTSBUFFERS_HH

struct ResultsBuffers {

    explicit ResultsBuffers(int s) : response(), size(s) {}

    ResultsBuffers(const ResultsBuffers &) = delete;

    ~ResultsBuffers() {}

    tbb::concurrent_queue<Response> response;
    int size;
};

#endif //KVCG_RESULTSBUFFERS_HH
