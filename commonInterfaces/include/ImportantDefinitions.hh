/**
 * @file
 */


#ifndef KVCG_IMPORTANTDEFINITIONS_HH
#define KVCG_IMPORTANTDEFINITIONS_HH

template<typename T>
struct EMPTY {
    static constexpr T value{};
};

template<typename T>
__forceinline__ __host__ __device__ unsigned compare(const T &lhs, const T &rhs);

#endif //KVCG_IMPORTANTDEFINITIONS_HH
