#pragma once

#include <stdlib.h>  
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <bitset>

#define DEFAULT_MISSILE_ADDR "localhost"
#define DEFAULT_MISSILE_PORT 34543

#ifndef RESOURCE_DIR
#define RESOURCE_DIR "../resource"
#endif

#define ASSERT(condition)\
     do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)

#define ASSERT_STATUS(cmd) ASSERT(cmd == Status::Succ)

#define RETURN_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        LOG(ERROR) << #cmd " error, " << __FILE__ << ":" << __LINE__; \
        return s;\
    }\
}

#define ASSERT_MSG(condition, message) \
    do { \
        if (! (condition)) { \
            LOG(ERROR) << "Assertion `" #condition "` failed in " << __FILE__ \
                      << ":" << __LINE__ << " msg: " << message; \
            std::terminate(); \
        } \
    } while (false)


namespace missilebase {

enum Status {
    Succ,
    Fail,
    NotFound,
    OutOfRange,
    InvalidArgument,
    Full,
    Timeout
};

template <typename T>
T align_up(T value, T alignment) {
    T temp = value % alignment;
    return temp == 0? value : value - temp + alignment;
}

template <typename T>
T align_down(T value, T alignment) {
    return value - value % alignment;
}

template <typename T>
T int_min(T a, T b){
    return a<b?a:b;
}

std::vector<std::string> split_by_delimiter(const std::string &s, char delim);

enum DeviceType {
    DeviceTypeTeslaV100,
    DeviceTypeTeslaP40,
    DeviceTypeRTX3060,
    DeviceTypeINVALIDDEVICE
};

DeviceType get_cuda_device_type(std::string device_name);

}

typedef std::bitset<64> GPUTPCAllocatedMap_t;
