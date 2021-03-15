//
// Created by depaulsmiller on 3/14/21.
//

#ifndef KVCG_STATDATA_CUH
#define KVCG_STATDATA_CUH

struct StatData {
    std::chrono::high_resolution_clock::time_point sampleQTime;
    std::chrono::high_resolution_clock::time_point timestampEnd;
    std::chrono::high_resolution_clock::time_point timestampWriteBack;
    std::chrono::high_resolution_clock::time_point timestampStartBatch;
    std::chrono::high_resolution_clock::time_point timestampDequeueToBatch;
    float duration;
    int size;
    int timesGoingToCache;
};

#endif //KVCG_STATDATA_CUH
