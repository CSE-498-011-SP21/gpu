#ifndef LISTALLOCATOR_HH
#define LISTALLOCATOR_HH

#include <iostream>

namespace groupallocator {

/**
 * Gets padding of a type
 */
    size_t getPadding(size_t startingptr, size_t alignment) {
        size_t multiplier = startingptr / alignment + 1;
        size_t padding = multiplier * alignment - startingptr;
        return padding;
    }

// when allocating
// write pointer to next
// then have the data

    struct MallocData {
        size_t size;
        size_t used;
        void* start;
    };

// not thread safe and no compaction
    class ListAllocator {
    public:
        // s needs to be larger than 2 MallocData
        ListAllocator(void *p, size_t s) : ptr(p), size(s) {
            // needs to maintain p to p + s
            l.push_back({s, 0, p});
        }

        ListAllocator() : ptr(nullptr), size(0) {}

        // allocates data in a free area or sets p to nullptr
        template<typename T>
        void alloc(T **p, size_t s, bool forceAligned128) {
            size_t alignment = forceAligned128 ? 128 : std::alignment_of<T *>();

            for(auto iter = l.begin(); iter != l.end(); ++iter) {

                if(iter->used == 0 && getPadding((size_t)iter->start, alignment) + s <= iter->size){

                    *p = (T*) iter->start;

                    size_t prevSize = iter->size;
                    void* prevStart = iter->start;

                    iter->size = s + getPadding((size_t)iter->start, alignment);
                    iter->used = 1;

                    MallocData m = {prevSize - iter->size, 0, (void*)((size_t)prevStart + iter->size)};
                    iter++;
                    l.insert(iter, m);
                    return;
                }
            }

            *p = nullptr;
        }

        // right now there is no compaction
        template<typename T>
        void free(T *p) {
            for(auto & iter : l){
                if((size_t)iter.start == (size_t)p){
                    iter.used = 0;
                    return;
                }
            }
        }

    private:
        void *ptr;
        size_t size;
        std::list<MallocData> l;

    };
}  // namespace groupallocator
#endif