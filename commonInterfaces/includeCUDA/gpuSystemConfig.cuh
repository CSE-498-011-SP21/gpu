/**
 * @file
 */

#ifndef KVCG_GPUSYSTEMCONFIG_CUH
#define KVCG_GPUSYSTEMCONFIG_CUH

/**
 * PartitionedSlabUnifiedConfig has 3 configurable parameters
 * size of the map, the gpu used, and the stream
 */
struct PartitionedSlabUnifiedConfig {
    int size;
    int gpu;
    cudaStream_t stream;
};

#endif //KVCG_GPUSYSTEMCONFIG_CUH
