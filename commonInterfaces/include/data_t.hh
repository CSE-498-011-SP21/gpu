/**
 * @file
 */

#ifndef KVCG_DATA_T_HH
#define KVCG_DATA_T_HH

struct data_t {

    data_t() : size(0), data(nullptr) {}

    data_t(size_t s) : size(s), data(new char[s]) {}

    /// Note this doesn't free the underlying data
    ~data_t() {}

    size_t size;
    char *data;

    inline data_t &operator=(const data_t &rhs) {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

    inline volatile data_t &operator=(const data_t &rhs) volatile {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

};

#endif //KVCG_DATA_T_HH
