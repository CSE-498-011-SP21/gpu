#ifndef GALLOCATOR_CLASSES_HH
#define GALLOCATOR_CLASSES_HH

#include <cmath>
#include <functional>
#include <mutex>
#include <vector>
#include <list>
#include <utility>

#include "ListAllocator.cuh"

namespace groupallocator {

    /**
     * Assert that CUDA returned successful
     * @param code
     * @param file
     * @param line
     * @param abort
     */
    inline void gpuAssert(cudaError_t code, const char *file, int line,
                          bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                    line);
            if (abort)
                exit(code);
        }
    }

    /**
     * Allocates memory in a page set by page_size.
     */
    class InPageAllocator {
    public:
        InPageAllocator() = delete;

        /**
         * Create the in page allocator.
         */
        InPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size) {
            gpuAssert(cudaMallocManaged((void **) &mem, PAGE_SIZE), __FILE__, __LINE__);
            l = ListAllocator(mem, PAGE_SIZE);
        }

        /**
         * Deletes in page allocator, cuda context must exist to do so.
         */
        ~InPageAllocator() { gpuAssert(cudaFree(mem), __FILE__, __LINE__); }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in IPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            m.lock();
            l.alloc(ptr, s, forceAligned128);
            m.unlock();
        }

        template<class T>
        void free(T *ptr) {
            m.lock();
            l.free(ptr);
            m.unlock();
        }

        bool contains(size_t ptr) {
            return ptr >= (size_t) mem && ptr < (size_t) mem + PAGE_SIZE;
        }

        void moveToDevice(int device, cudaStream_t stream) {
            gpuAssert(cudaMemPrefetchAsync(mem, PAGE_SIZE, device, stream), __FILE__, __LINE__);
        }

        size_t getPages() { return 1; }

        size_t getPageSize() { return PAGE_SIZE; }

    private:
        char *mem;
        ListAllocator l;
        std::mutex m;
        const size_t PAGE_SIZE;
    };

    /**
     * Allocates multiple pages of memory
     */
    class MultiPageAllocator {
    public:
        MultiPageAllocator() = delete;

        /**
         * Constructor
         */
        MultiPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size), pagesAllocated(0) {}

        /**
         * Delete function
         */
        ~MultiPageAllocator() {
            m.lock();
            for (auto &e : mem) {
                gpuAssert(cudaFree((void *) e.first), __FILE__, __LINE__);
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in MPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            size_t pages_needed = (size_t) ceil(s / (double) PAGE_SIZE);
            char *c;
            gpuAssert(cudaMallocManaged((void **) &c, pages_needed * PAGE_SIZE), __FILE__, __LINE__);
            *ptr = (T *) c;
            m.lock();
            pagesAllocated += pages_needed;
            mem.push_back({c, pages_needed * PAGE_SIZE});
            m.unlock();
        }

        template<class T>
        void free(T *ptr) {
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                if ((size_t) i->first == (size_t) ptr) {
                    gpuAssert(cudaFree((void *) i->first), __FILE__, __LINE__);
                    mem.erase(i);
                    break;
                }
            }
            m.unlock();
        }

        void moveToDevice(int device, cudaStream_t stream) {
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                gpuAssert(cudaMemPrefetchAsync(i->first, i->second, device, stream), __FILE__, __LINE__);
            }
            m.unlock();
        }

        size_t getPages() {
            std::unique_lock <std::mutex> ul(m);
            return pagesAllocated;
        }

        size_t getPageSize() { return PAGE_SIZE; }

    private:
        std::list <std::pair<char *, size_t>> mem;
        std::mutex m;
        const size_t PAGE_SIZE;
        size_t pagesAllocated;
    };

    /**
     * Allocates with group affinity
     */
    class GroupAllocator {
    public:

        GroupAllocator() = delete;

        /**
         * Constructor takes group_num to allocate to
         * @param group_num
         */
        GroupAllocator(int group_num, std::size_t page_size)
                : group_num_(group_num),
                  PAGE_SIZE(page_size) {
            mpa = new MultiPageAllocator(page_size);
        }

        /**
         * Delete function
         */
        ~GroupAllocator() {}

        /**
         * Function to free all memory of group allocator
         */
        void freeall() {
            for (auto &e : ipas) {
                delete e;
            }
            delete mpa;
        }

        /**
         * Free pointer T* ptr
         * @tparam T
         * @param ptr
         */
        template<class T>
        void free(T *ptr) {
            mpa->free(ptr);
            m.lock();
            for (auto &e : ipas) {
                if (e->contains((size_t) ptr)) {
                    e->free(ptr);
                    break;
                }
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @tparam T
         * @param ptr
         * @param s
         * @param forceAligned128
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
            if (ptr == NULL || s == 0) {
                return;
            }

            if (s + alignof(T *) > PAGE_SIZE) {
                mpa->allocate<T>(ptr, s, forceAligned128);
            } else {
                m.lock();
                int lastSize = ipas.size();
                if (lastSize == 0) {
                    InPageAllocator *ipa_new =
                            new InPageAllocator(PAGE_SIZE);
                    ipas.push_back(ipa_new);
                }
                auto ipa3 = ipas[ipas.size() - 1];
                m.unlock();
                ipa3->allocate<T>(ptr, s, forceAligned128);
                while (*ptr == NULL) {
                    InPageAllocator *ipa2 =
                            new InPageAllocator(PAGE_SIZE);
                    m.lock();
                    if (lastSize == ipas.size()) {
                        ipas.push_back(ipa2);
                        lastSize = ipas.size();
                    }
                    m.unlock();
                    m.lock();
                    auto ipa = ipas[ipas.size() - 1];
                    m.unlock();
                    ipa->allocate<T>(ptr, s, forceAligned128);
                }
            }
        }

        /**
         * Move to memory device in stream
         * @param device
         * @param stream
         */
        void moveToDevice(int device, cudaStream_t stream) {
            mpa->moveToDevice(device, stream);
            m.lock();
            for (auto &e : ipas) {
                e->moveToDevice(device, stream);
            }
            m.unlock();
        }

        size_t pagesAllocated() {
            auto s = mpa->getPages();
            m.lock();
            s += ipas.size();
            m.unlock();
            return s;
        }

        size_t getPageSize() {
            return PAGE_SIZE;
        }

    private:
        std::vector<InPageAllocator *> ipas;
        MultiPageAllocator *mpa;
        std::mutex m;
        int group_num_;
        const size_t PAGE_SIZE;
    };

}  // namespace groupallocator

#endif