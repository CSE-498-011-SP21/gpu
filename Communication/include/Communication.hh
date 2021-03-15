/**
 * @file
 */

#include <data_t.hh>
#include <Messages.hh>
#include <tbb/concurrent_queue.h>

#ifndef KVCG_COMMUNICATION_HH
#define KVCG_COMMUNICATION_HH

struct Communication {

    explicit Communication(int s) : response(), size(s) {
        response.set_capacity(s);
    }

    Communication(const Communication &) = delete;

    ~Communication() {}

    void send(Response &&r) {
        response.push(r);
    }

    bool try_recv(Response &r) {
        return response.try_pop(r);
    }

    int size;
private:
    tbb::concurrent_bounded_queue<Response> response;
};

#endif //KVCG_RESULTSBUFFERS_HH
