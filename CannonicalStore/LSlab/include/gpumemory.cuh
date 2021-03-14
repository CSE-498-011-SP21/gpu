#include "gpuErrchk.cuh"
#include <iostream>

#ifndef GPUMEMORY_CUH
#define GPUMEMORY_CUH

template<typename T>
struct GPUCPUMemory {

    GPUCPUMemory() : host(nullptr), size(0), device(nullptr) {
    }

    GPUCPUMemory(size_t size) : GPUCPUMemory(new T[size], size) {}

    GPUCPUMemory(T *h, size_t size) : host(h), size(size), device(new T *[1]) {
        gpuErrchk(cudaMalloc(&device[0], sizeof(T) * size))
    }

    GPUCPUMemory(GPUCPUMemory<T> &&ref) noexcept {
        host = ref.host;
        device = ref.device;
        size = ref.size;
        ref.host = nullptr;
        ref.device = nullptr;
    }

    ~GPUCPUMemory() {
        if (device != nullptr) {
            std::cerr << "Deleting memory\n";
            gpuErrchk(cudaFree(*device))
            delete[] device;
        }
    }

    inline GPUCPUMemory<T> &operator=(GPUCPUMemory<T> &&other) {
        if (&other != this) {
            if (device != nullptr) {
                gpuErrchk(cudaFree(*device))
                delete[] device;
            }
            host = other.host;
            device = other.device;
            size = other.size;
            other.host = nullptr;
            other.device = nullptr;
        }
        return *this;
    }

    inline void movetoGPU() {
        gpuErrchk(
                cudaMemcpy(*device, host, sizeof(T) * size, cudaMemcpyHostToDevice))
    }

    inline void movetoCPU() {
        gpuErrchk(
                cudaMemcpy(host, *device, sizeof(T) * size, cudaMemcpyDeviceToHost))
    }

    inline T *getDevice() {
        return *device;
    }

    T *host;
    size_t size;
private:
    T **device;
};

#endif