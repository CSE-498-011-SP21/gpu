/**
 * @file
 */

#include "data_t.hh"

#ifndef KVCG_MESSAGES_HH
#define KVCG_MESSAGES_HH

struct Response {

    Response(int id, data_t *res, bool ret) : requestID(id), result(res), retry(ret) {

    }

    Response() : requestID(-1) {

    }


    ~Response() {}

    int requestID;
    data_t *result;
    bool retry;
};

#endif //KVCG_MESSAGES_HH
