/** @file GroupAllocator.cuh
 * Allocation and deallocation functionality resides here.
 */
#ifndef GALLOCATOR_HH
#define GALLOCATOR_HH

#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "GroupAllocator_Classes.cuh"

namespace groupallocator {

    static std::mutex groupMapMutex;
    static std::unordered_map<int, std::shared_ptr<GroupAllocator>> allocator;

    /**
     * Context object
     */
    struct Context {
        /**
         * Create context
         */
        Context() {}
        /**
         * Delete context
         */
        ~Context() {}
        /**
         * Size of pages to use
         */
        std::size_t page_size = 4096;
    };

    /**
     * Allocates memory of type T and sets *ptr to this memory of size s. It
     * allocates in group group.
     * Thread Safe!
     * @param ptr
     * @param s
     * @param group
     */
    template<typename T>
    void allocate(T **ptr, size_t s, const Context ctx, int group = -1, bool forceAligned128 = false) {
        groupMapMutex.lock();
        std::shared_ptr <GroupAllocator> g = allocator[group];
        if (g == nullptr) {
            g = std::make_shared<GroupAllocator>(group, ctx.page_size);
            allocator[group] = g;
        }
        groupMapMutex.unlock();

        g->allocate<T>(ptr, s, forceAligned128);
    }

    /**
     * Free T* p from group
     * @tparam T
     * @param p
     * @param group
     */
    template<typename T>
    void free(T *p, int group = -1) {
        groupMapMutex.lock();
        std::shared_ptr <GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            g->free(p);
        }
    }

    /**
     * Cleans up the allocator by freeing everything so there is no memory leak.
     * Thread safe.
     */
    void freeall() {
        groupMapMutex.lock();
        for (std::pair<const int, std::shared_ptr < groupallocator::GroupAllocator>>
            &elm : allocator) {
            elm.second->freeall();
        }
        groupMapMutex.unlock();
    }

    /**
     * Moves data to GPU, thread safe
     * @param group
     * @param gpuID
     * @param stream
     */
    void moveToGPU(int group = -1, int gpuID = 0, cudaStream_t stream = cudaStreamDefault){
        groupMapMutex.lock();
        std::shared_ptr <GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            //std::cerr << "Using stream " << stream << std::endl;

            g->moveToDevice(gpuID, stream);
        }
    }

    /**
     * Moves data to CPU, thread safe
     * @param group
     * @param stream
     */
    void moveToCPU(int group = -1, cudaStream_t stream = cudaStreamDefault){
        groupMapMutex.lock();
        std::shared_ptr <GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            //std::cerr << "Using stream " << stream << std::endl;
            g->moveToDevice(cudaCpuDeviceId, stream);
        }
    }

}  // namespace groupallocator
#endif