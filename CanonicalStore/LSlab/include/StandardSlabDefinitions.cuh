//
// Created by depaulsmiller on 8/28/20.
//
#include <Operations.cuh>
#include <functional>
#include <data_t.hh>

#ifndef LSLABEXT_STANDARDSLABS_CUH
#define LSLABEXT_STANDARDSLABS_CUH

/// For use with shared_ptr
class Data_tDeleter{
    inline void operator()(data_t* ptr) const noexcept {
        delete[] ptr->data;
        delete ptr;
    }
};

template<>
struct EMPTY<data_t *> {
    static constexpr data_t *value = nullptr;
};

template<>
__forceinline__ __device__ unsigned compare(data_t *const &lhs, data_t *const &rhs) {

    if (lhs == rhs) {
        return 0;
    } else if (lhs == nullptr || rhs == nullptr) {
        return 1;
    }

    if (lhs->size != rhs->size) {
        return (unsigned) (lhs->size - rhs->size);
    }

    for (size_t i = 0; i < lhs->size; i++) {
        unsigned sub = lhs->data[i] - rhs->data[i];
        if (sub != 0)
            return sub;
    }

    return 0;
}

namespace std {
    template<>
    struct std::hash<data_t *> {
        inline std::size_t operator()(data_t *&x) {
            return std::hash<std::string>{}(x->data) ^ std::hash<std::size_t>{}(x->size);
        }
    };
}

template<>
struct EMPTY<unsigned> {
    static const unsigned value = 0;
};

template<>
__forceinline__ __device__ unsigned compare(const unsigned &lhs, const unsigned &rhs) {
    return lhs - rhs;
}

template<>
__forceinline__ __device__ unsigned compare(const unsigned long long &lhs, const unsigned long long &rhs) {
    return lhs - rhs;
}


#endif //LSLABEXT_STANDARDSLABS_CUH
