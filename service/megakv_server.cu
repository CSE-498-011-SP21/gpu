//
// Created by depaulsmiller on 1/15/21.
//

#include <unistd.h>
#include <MegaKV.cuh>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <dlfcn.h>
#include <tbb/pipeline.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/concurrent_vector.h>

namespace pt = boost::property_tree;

int totalBatches = 10000;
int BATCHSIZE = 512;
int NUM_THREADS = std::thread::hardware_concurrency() - 4;
using time_point = std::chrono::high_resolution_clock::time_point;

void usage(char *command);

double stage1(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
              megakv::GPUData *data);

double
stage3(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
       megakv::GPUData *data,
       std::shared_ptr<megakv::Response> resp);

static void pipeline_exec(megakv::MegaKV<std::string, std::string> *s, int nstreams, std::atomic_bool &done,
                          std::atomic_long &processed,
                          tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>> *q,
                          tbb::concurrent_vector<std::pair<time_point, double>> &times,
                          time_point &end);

void (*initWorkload)() = nullptr;

void (*initWorkloadFile)(std::string) = nullptr;

std::shared_ptr<megakv::BatchOfRequests> (*generateWorkloadBatch)(unsigned int *, unsigned) = nullptr;

std::vector<std::shared_ptr<megakv::BatchOfRequests>> (*getPopulationBatches)(unsigned *, unsigned) = nullptr;

int (*getBatchesToRun)() = nullptr;

struct ServerConf {
    int threads;
    int nstreams;
    int size;

    ServerConf() {
        threads = NUM_THREADS;
        nstreams = 10;
        size = 1000000;
    }

    ServerConf(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        threads = root.get<int>("threads", NUM_THREADS);
        nstreams = root.get<int>("nstreams", 10);
        size = root.get<int>("size", 10000000);
    }

    void persist(std::string filename) {
        pt::ptree root;
        root.put("threads", threads);
        root.put("nstreams", 10);
        root.put("size", size);
        pt::write_json(filename, root);
    }

    ~ServerConf() {

    }

};

int main(int argc, char **argv) {

    ServerConf sconf;

    bool workloadFilenameSet = false;
    std::string workloadFilename = "";
    std::string dllib = "./libmkvzipfianWorkload.so";

    char c;
    while ((c = getopt(argc, argv, "f:w:l:")) != -1) {
        switch (c) {
            case 'f':
                sconf = ServerConf(std::string(optarg));
                // optarg is the file
                break;
            case 'w':
                workloadFilenameSet = true;
                workloadFilename = std::string(optarg);
                break;
            case 'l':
                dllib = std::string(optarg);
                break;
            case '?':
                usage(argv[0]);
                return 1;
        }
    }

    NUM_THREADS = sconf.threads;
    auto *s = new megakv::MegaKV<std::string, std::string>(sconf.size);

    auto handler = dlopen(dllib.c_str(), RTLD_LAZY);
    if (!handler) {
        std::cerr << dlerror() << std::endl;
        return 1;
    }
    initWorkload = (void (*)()) dlsym(handler, "initWorkload");
    initWorkloadFile = (void (*)(std::string)) dlsym(handler, "initWorkloadFile");
    generateWorkloadBatch = (std::shared_ptr<megakv::BatchOfRequests>(*)(unsigned *, unsigned)) dlsym(handler,
                                                                                                      "generateWorkloadBatch");
    getPopulationBatches = (std::vector<std::shared_ptr<megakv::BatchOfRequests>>(*)(unsigned *, unsigned)) dlsym(
            handler, "getPopulationBatches");
    getBatchesToRun = (int (*)()) dlsym(handler, "getBatchesToRun");

    if (workloadFilenameSet) {
        initWorkloadFile(workloadFilename);
    } else {
        initWorkload();
    }

    std::cerr << "Starting populating" << std::endl;

    unsigned seed = time(nullptr);
    auto popBatches = getPopulationBatches(&seed, BATCHSIZE);
    auto tmpData = new megakv::GPUData();
    auto batchesIter = popBatches.begin();
    while (batchesIter != popBatches.end()) {
        std::vector<std::shared_ptr<megakv::BatchOfRequests>> batchToRun;
        for (int i = 0; i < megakv::BLOCKS && batchesIter != popBatches.end(); i++) {
            batchToRun.push_back(*batchesIter);
            ++batchesIter;
        }
        auto resp = std::make_shared<megakv::Response>(megakv::BLOCKS * megakv::THREADS_PER_BLOCK);
        s->preprocess_hashes(batchToRun, tmpData);
        s->preprocess_rest(batchToRun, tmpData);
        s->moveTo(tmpData, cudaStreamDefault);
        s->execute(tmpData, cudaStreamDefault);
        s->moveFrom(tmpData, cudaStreamDefault);
        cudaStreamSynchronize(cudaStreamDefault);
        s->postprocess(batchToRun, resp, tmpData);
    }
    delete tmpData;

    std::cerr << "Starting workload" << std::endl;

    auto *q = new tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>>();
    std::atomic_bool done = false;


    int batches = getBatchesToRun();
    std::atomic_long processed{0};


    tbb::concurrent_vector<std::pair<time_point, double>> times;
    times.reserve(batches);
    time_point end;

    auto f = std::thread([&]() {
        pipeline_exec(s, sconf.nstreams, done, processed, q, times, end);
    });

    auto start = std::chrono::high_resolution_clock::now();

    const int clients = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < clients; i++) {
        threads.push_back(std::thread([&](int tid) {
            unsigned s = time(nullptr);
            for (int j = 0; j < batches / clients; j++) {
                q->push(std::move(generateWorkloadBatch(&s, megakv::THREADS_PER_BLOCK)));
            }
            if (tid == 0) {
                for (int j = 0; j < batches - (batches / clients) * clients; j++) {
                    q->push(std::move(generateWorkloadBatch(&s, megakv::THREADS_PER_BLOCK)));
                }
            }
        }, i));
    }

    for (auto &t : threads) {
        t.join();
    }
    auto end_arrival = std::chrono::high_resolution_clock::now();

    done = true;

    f.join();

    std::sort(times.begin(), times.end(),
              [](const std::pair<time_point, double> &lhs, const std::pair<time_point, double> &rhs) {
                  return lhs.first < rhs.first;
              });

    std::vector<std::pair<double, double>> times_processed;
    for (auto &t: times) {
        times_processed.emplace_back(std::chrono::duration<double>(t.first - start).count(), t.second * 1e3);
    }

    std::cout << "TABLE: Latency" << std::endl;

    std::cout << "Time\tLatency" << std::endl;
    for (auto &t : times_processed) {
        std::cout << t.first << "\t" << t.second << std::endl;
    }
    std::cout << std::endl;

    std::cout << "TABLE: Throughput" << std::endl;
    std::cout << "Throughput" << std::endl;
    std::cout << processed.load() * megakv::THREADS_PER_BLOCK /
                 std::chrono::duration<double>(end - start).count() / 1e6
              << std::endl;
    std::cout << std::endl;

    std::cerr << "Throughput: " << processed.load() * megakv::THREADS_PER_BLOCK /
                                   std::chrono::duration<double>(end - start).count() / 1e6
              << std::endl;
    std::cerr << "Arrival: " << processed.load() * megakv::THREADS_PER_BLOCK /
                                std::chrono::duration<double>(end_arrival - start).count() / 1e6
              << std::endl;

    delete s;
    delete q;

    dlclose(handler);

    gpuErrchk(cudaDeviceReset());

    return 0;
}

void usage(char *command) {
    using namespace std;
    cout << command << " [-f <config file>]" << std::endl;
}

double stage1(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
              megakv::GPUData *data) {
    auto start = std::chrono::high_resolution_clock::now();
    s->preprocess_hashes(reqs, data);
    s->preprocess_rest(reqs, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

double
stage3(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
       megakv::GPUData *data,
       std::shared_ptr<megakv::Response> resp) {
    auto start = std::chrono::high_resolution_clock::now();
    s->postprocess(reqs, resp, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

static void pipeline_exec(megakv::MegaKV<std::string, std::string> *s, int nstreams, std::atomic_bool &done,
                          std::atomic_long &processed,
                          tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>> *q,
                          tbb::concurrent_vector<std::pair<time_point, double>> &times,
                          time_point &end) {

    using vec_type = std::vector<std::shared_ptr<megakv::BatchOfRequests>>;

    megakv::GPUData **data = new megakv::GPUData *[nstreams];
    for (int i = 0; i < nstreams; i++) {
        data[i] = new megakv::GPUData();
    }

    std::vector<std::shared_ptr<megakv::Response>> resp;
    for (int i = 0; i < nstreams; i++) {
        resp.push_back(std::make_shared<megakv::Response>(megakv::BLOCKS * megakv::THREADS_PER_BLOCK));
    }
    cudaStream_t streams[nstreams];

    for (int i = 0; i < nstreams; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

    assert(times.empty());

    std::vector<std::thread> streamsToRun;
    for (int stream = 0; stream < nstreams; stream++) {
        streamsToRun.emplace_back([&](int tid) {
            while (!done) {
                auto v = vec_type();
                int i = 0;
                for (; i < megakv::BLOCKS &&
                       !done; i++) {
                    std::shared_ptr<megakv::BatchOfRequests> b;
                    if (q->try_pop(b)) {
                        v.push_back(std::move(b));
                        processed.fetch_add(1,
                                            std::memory_order_relaxed);
                    }
                }
                if (!v.empty()) {
                    auto start = std::chrono::high_resolution_clock::now();
                    stage1(s, v, data[tid]);
                    s->moveTo(data[tid], streams[tid]);
                    s->execute(data[tid], streams[tid]);
                    s->moveFrom(data[tid], streams[tid]);
                    cudaStreamSynchronize(streams[tid]);
                    stage3(s, v,
                           data[tid],
                           resp[tid]);
                    double dur = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - start).count();

                    times.push_back({start, dur});
                }
            }
        }, stream);
    }

    for (auto &t: streamsToRun) {
        t.join();
    }
    end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nstreams; i++) {
        delete data[i];
    }

    delete[] data;

    for (int i = 0; i < nstreams; i++) gpuErrchk(cudaStreamDestroy(streams[i]));


}