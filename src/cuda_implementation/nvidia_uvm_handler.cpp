//
// Created by Anonymous authors on 2023/7/23.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <dlfcn.h>
#include <iomanip>
#include <sys/ioctl.h>
#include <string>
#include <nvml.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "cuda_impl.h"
#include "nvidia_uvm_handler.h"

#define NVIDIA_UVM_DEVICE_NAME          "nvidia-uvm"
#define NVIDIA_UVM_DEVICE_PATH  "/dev/" NVIDIA_UVM_DEVICE_NAME

/* Aligns fields in structs  so they match up between 32 and 64 bit builds */
#if defined(__GNUC__) || defined(__clang__) || defined(NV_QNX) || defined(NV_HOS)
#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))
#elif defined(__arm)
#define NV_ALIGN_BYTES(size) __align(ALIGN)
#else
// XXX This is dangerously nonportable!  We really shouldn't provide a default
// version of this that doesn't do anything.
#define NV_ALIGN_BYTES(size)
#endif

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

#define NV_UUID_LEN 16

typedef struct nv_uuid
{
    uint8_t uuid[NV_UUID_LEN];
} NvUuid;

typedef NvUuid NvProcessorUuid;

#define UVM_GET_PHYS_ADDR                                             UVM_IOCTL_BASE(76)
typedef struct
{
    uint64_t          virt_addr      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid uuid;                             // IN
    uint64_t           phys_addr      NV_ALIGN_BYTES(8); // OUT
    uint32_t      rmStatus;                         // OUT
} UVM_GET_PHYS_ADDR_PARAMS;

#define IOCTL_UVM_GET_PHYS_ADDR            _IOC(0, 0, UVM_GET_PHYS_ADDR, 0)

typedef int (*orig_open_f_type)(const char *pathname, int flags, int mode);
orig_open_f_type g_orig_open;

pthread_once_t g_pre_init_once = PTHREAD_ONCE_INIT;
pthread_once_t g_post_init_once = PTHREAD_ONCE_INIT;
bool g_init_failed;

/* Retrieve the device UUID from the CUDA device handle */
static int get_device_UUID(int device, NvProcessorUuid *uuid)
{
    nvmlReturn_t ncode;
    cudaError_t ccode;
    char pciID[32];
    nvmlDevice_t handle;
    char buf[100];
    char hex[3];
    char *nbuf;
    int cindex, hindex, uindex, needed_bytes;
    char c;
    int len;
    std::string prefix = "GPU";
    const char *gpu_prefix = prefix.c_str();
    int gpu_prefix_len = strlen(gpu_prefix);

    ncode = nvmlInit();
    if (ncode != NVML_SUCCESS){
        LOG(ERROR) << "FGPU: Failed to init the NVML library, err = " << ncode;
        return -EINVAL;
    }

    ncode = nvmlDeviceGetHandleByIndex(device, &handle);
    if (ncode != NVML_SUCCESS){
        LOG(ERROR) << "FGPU:Couldn't get Device Handle, err = " << ncode;
        return -EINVAL;
    }


    ncode = nvmlDeviceGetUUID(handle, buf, sizeof(buf));
    if (ncode != NVML_SUCCESS){
        LOG(ERROR) << "FGPU:Couldn't find device UUID, err = " << ncode;
        return -EINVAL;
    }

    if (strncmp(buf, gpu_prefix, gpu_prefix_len != 0))
        return 0;

    nbuf = buf + gpu_prefix_len;

    /*
     * UUID has characters and hexadecimal numbers.
     * We are only interested in hexadecimal numbers.
     * Each hexadecimal numbers is equal to 1 byte.
     */
    needed_bytes = sizeof(NvProcessorUuid);
    len = strlen(nbuf);

    for (cindex = 0, hindex = 0, uindex = 0; cindex < len; cindex++) {
        c = nbuf[cindex];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            hex[hindex] = c;
            hindex++;
            if (hindex == 2) {
                hex[2] = '\0';
                uuid->uuid[uindex] = (uint8_t)strtol(hex, NULL, 16);
                uindex++;
                hindex = 0;
                if (uindex > needed_bytes) {
                    LOG(ERROR) << "FGPU:Invalid device UUID";
                    return -EINVAL;
                }
            }
        }
    }

    if (uindex != needed_bytes) {
        LOG(ERROR) << "FGPU:Invalid device UUID";
        return -EINVAL;
    }

    return 0;
}

/* Does the most neccesary initialization */
static void pre_initialization(void)
{
    g_orig_open = (orig_open_f_type)dlsym(RTLD_NEXT,"open");
    if (!g_orig_open) {
        g_init_failed = true;
        return;
    }
}

static void post_initialization(void)
{
    // TODO: Now do nothing.
    /*nvmlReturn_t ncode;

    ncode = nvmlInit();
    if (ncode != NVML_SUCCESS) {
        g_init_failed = true;
        return;
    }*/
}

/* Does the initialization atmost once */
static int init(bool do_post_init)
{
    int ret;

    ret = pthread_once(&g_pre_init_once, pre_initialization);
    if (ret < 0)
        return ret;

    if (g_init_failed) {
        LOG(ERROR) << "FGPU:Initialization failed";
        return -EINVAL;
    }

    if (!do_post_init)
        return 0;

    ret = pthread_once(&g_post_init_once, post_initialization);
    if (ret < 0)
        return ret;

    if (g_init_failed) {
        LOG(ERROR) << "FGPU:Initialization failed";
        return -EINVAL;
    }

    return 0;
}

extern "C" {

/* Trap open() calls (interested in UVM device opened by CUDA) */
int open(const char *pathname, int flags, int mode) {
    int ret;

    ret = init(false);
    if (ret < 0)
        return ret;

    ret = g_orig_open(pathname, flags, mode);

    if (g_uvm_fd < 0 &&
        strncmp(pathname, NVIDIA_UVM_DEVICE_PATH, strlen(NVIDIA_UVM_DEVICE_PATH)) == 0) {
        g_uvm_fd = ret;
    }

    return ret;
}
}

void get_gpu_uuid(int gpu_id, uint8_t *uuid) {
    CUuuid* cuuuid = new(CUuuid);
    CUdevice device;
    ASSERT_GPU_ERROR(cuDeviceGet(&device, gpu_id));
    ASSERT_GPU_ERROR(cuDeviceGetUuid(cuuuid, device));
    for(int i=0; i<16; i++) {
        uuid[i] = cuuuid->bytes[i];
    }
    delete(cuuuid);
}

int get_phys_addr_from_gpu_virt_addr(int gpu_id, void* virt_addr, void** phys_addr) {
    UVM_GET_PHYS_ADDR_PARAMS params;
    params.virt_addr = (uint64_t) virt_addr;
    int ret;

    uint8_t uuid_array[20];
    get_gpu_uuid(gpu_id, params.uuid.uuid);

    ret = ioctl(g_uvm_fd, IOCTL_UVM_GET_PHYS_ADDR, &params);
    if (ret < 0)
        return ret;

    if (params.rmStatus != 0) {
        LOG(ERROR) << "Couldn't get gpu physical address, retval=" << params.rmStatus;
        return -EINVAL;
    }
    *phys_addr = (void*)(((size_t)(params.phys_addr) & (size_t)((1llu<<33llu)-(1llu<<8llu)))<<4llu);
    return 0;
}