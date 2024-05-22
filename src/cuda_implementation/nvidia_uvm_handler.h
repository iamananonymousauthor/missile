//
// Created by Anonymous authors on 2023/7/23.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <string>
#include <nvml.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

static int g_uvm_fd = -1;

void get_gpu_uuid(int gpu_id, uint8_t *uuid);

int open_nvidia_uvm(const char *pathname, int flags, int mode);

int get_phys_addr_from_gpu_virt_addr(int gpu_id, void* virt_addr, void** phys_addr);