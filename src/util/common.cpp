#include <stdlib.h>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "util/common.h"

namespace missilebase {

std::vector<std::string> split_by_delimiter(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

DeviceType get_cuda_device_type(std::string device_name) {
    DeviceType gpu_type = DeviceTypeINVALIDDEVICE;
    for(int i=0; i+1<device_name.length(); i++) {
        if(device_name[i] == 'V' && device_name[i+1] == '1') { //V100
            gpu_type = DeviceTypeTeslaV100;
            break;
        } else if(device_name[i] == 'P' && device_name[i+1] == '4') { //P40
            gpu_type = DeviceTypeTeslaP40;
            break;
        } else if(device_name[i] == '3' && device_name[i+1] == '0') { //3060
            gpu_type = DeviceTypeRTX3060;
            break;
        }
    }
    ASSERT_MSG(gpu_type != DeviceTypeINVALIDDEVICE, "Invalid GPU type: " << device_name);
    return gpu_type;
}

}

