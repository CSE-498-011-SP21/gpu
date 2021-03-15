#include <KVCache.hh>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>
#include <random>

using cache_t = kvgpu::KVCache<unsigned long long, unsigned long long, 10000000>;

struct barrier_t {
    std::condition_variable cond;
    std::mutex mtx;
    int count;
    int crossing;

    barrier_t(int n) : count(n), crossing(0) {}

    void wait() {
        std::unique_lock<std::mutex> ulock(mtx);
        /* One more thread through */
        crossing++;
        /* If not all here, wait */
        if (crossing < count) {
            cond.wait(ulock);
        } else {
            cond.notify_all();
            /* Reset for next time */
            crossing = 0;
        }
    }
};

void thread_func(barrier_t &b, int ops, int range,
                 cache_t &cache, std::vector<int> &foundValid, std::vector<int> &searchTotal, int tid) {

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned long long> distribution(1, range);

    kvgpu::SimplModel<unsigned long long> m;
    b.wait();
    for (int w = 0; w < ops; w++) {
        unsigned long long k = distribution(generator);
        auto ret = cache.fast_get(k, k, m);
        if (ret.first && ret.first->valid == 1) {
            foundValid[tid] = foundValid[tid] + 1;
        }
        searchTotal[tid] = searchTotal[tid] + 1;
    }
}

int main(int argc, char **argv) {
    int nthreads = 8;

    if (argc > 1) {
        nthreads = atoi(argv[1]);
    }
    int ops = 1000000;

    cache_t cache;
    std::vector<std::thread> threads;

    barrier_t b(nthreads + 1);

    std::vector<int> foundValid(nthreads);
    std::vector<int> searchTotal(nthreads);

    int range = 1000000;

    for (int i = 0; i < range; i++) {
        kvgpu::SimplModel<unsigned long long> m;
        size_t log_loc;
        bool logLocSet;
        auto ret = cache.get_with_log(i, i, m);
        assert(std::get<0>(ret) != nullptr);
        std::get<0>(ret)->valid = 1;
        std::get<0>(ret)->key = (unsigned long long) i;
        std::get<0>(ret)->value = 2;
    }

    for (int i = 0; i < nthreads; i++) {
        threads.push_back(std::thread([&](int tid) {
            thread_func(b, ops, range, cache, foundValid, searchTotal, tid);
        }, i));
    }

    auto start = std::chrono::high_resolution_clock::now();
    b.wait();

    for (auto &t : threads) {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end - start).count() / ops * 1e6 << " us" << std::endl;
    std::cout << ops * nthreads / std::chrono::duration<double>(end - start).count() / 1e6 << " mops" << std::endl;

}