#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <bitset>
#include "util/common.h"

#define GPUInit cuInit
#define GPUDeviceGet cuDeviceGet
#define GPUDeviceGetCount cuDeviceGetCount
#define GPUDeviceGetName cuDeviceGetName
#define GPUSetDevice cudaSetDevice
#define GPUCtxCreate cuCtxCreate
#define GPUCtxSetCurrent cuCtxSetCurrent
#define GPUModuleLoad cuModuleLoad
#define GPUModuleGetFunction cuModuleGetFunction
#define GPUMalloc cuMemAlloc
#define GPUHostMalloc cuMemHostAlloc
#define GPUMemHostGetDevicePointer cuMemHostGetDevicePointer
#define GPUMemcpyHtoD cuMemcpyHtoD
#define GPUMemcpyDtoH cuMemcpyDtoH
#define GPUMemcpyHtoDAsync cuMemcpyHtoDAsync
#define GPUMemcpyDtoHAsync cuMemcpyDtoHAsync
#define GPUModuleLaunchKernel cuLaunchKernel
#define GPUStreamDefault 0
#define GPUStreamSynchronize cuStreamSynchronize
#define GPUStreamCreate(stream) cuStreamCreate((stream), CU_STREAM_NON_BLOCKING)
#define GPUStreamCreateWithPriority(stream, priority) cuStreamCreateWithPriority((stream), CU_STREAM_NON_BLOCKING, (priority))
#define GPUStreamQuery cuStreamQuery
#define GPUStatusOK CUDA_SUCCESS
#define GPUFree cuMemFree
#define GPUWriteValue32Async(stream, dstDevice, value, flag) cuMemcpyHtoDAsync((dstDevice), new int(value), sizeof(int), (stream))
#define GPUClearHostQueue cuStreamClearQueue
#define GPUResetCU missilebase::executor::cuResetWavefronts
#define GPUMemset cuMemsetD8
#define GPUMemset cuMemsetD8
#define GPUDeviceSynchronize cudaDeviceSynchronize
#define GPUBindTexture cudaBindTexture

#define GPUHostMallocDefault cudaHostAllocDefault

// #define GPU_RETURN_STATUS(cmd) return Status::Fail;

#define GPU_RETURN_STATUS(cmd)                                                               \
    {                                                                                        \
        CUresult error = cmd;                                                                \
        if (error != CUDA_SUCCESS)                                                           \
        {                                                                                    \
            const char *str;                                                                 \
            cuGetErrorString(error, &str);                                                   \
            std::string err_str(str);                                                        \
            LOG(INFO) << "cuda error: " << err_str << " at " << __FILE__ << ":" << __LINE__; \
            return Status::Fail;                                                             \
        }                                                                                    \
    }

#define GPU_CALL_RUNTIME_RETURN_STATUS(cmd)                                                                   \
    {                                                                                                         \
        cudaError_t error = cmd;                                                                              \
        if (error != CUDA_SUCCESS)                                                                            \
        {                                                                                                     \
            printf("DEBUG: Failed to call runtime cmd\n");                                                    \
            LOG(INFO) << "cuda error: " << cudaGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
            return Status::Fail;                                                                              \
        }                                                                                                     \
    }

#define ASSERT_GPU_ERROR(cmd)                                                                                             \
    {                                                                                                                     \
        CUresult error = cmd;                                                                                             \
        if (error != CUDA_SUCCESS)                                                                                        \
        {                                                                                                                 \
            const char *str;                                                                                              \
            cuGetErrorString(error, &str);                                                                                \
            std::string err_str(str);                                                                                     \
            LOG(ERROR) << "cuda error: " << err_str << "at " << __FILE__ << ":" << __LINE__ << ", error num = " << error; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

#define CALL_RUNTIME_FUNCTION(cmd)                                                                             \
    {                                                                                                          \
        cudaError_t error = cmd;                                                                               \
    if (error != CUDA_SUCCESS)                                                                             \
        {                                                                                                      \
            LOG(ERROR) << "cuda error: " << cudaGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    }

/*struct dim3 {
    dim3(){}
    dim3(unsigned int _x, unsigned int _y, unsigned int _z) : x(_x), y(_y), z(_z) {}
    unsigned int x;
    unsigned int y;
    unsigned int z;
};*/

namespace missilebase
{
    namespace executor
    {

        typedef CUdeviceptr GPUDevicePtr_t;
        typedef CUfunction GPUFunction_t;
        typedef CUdevice GPUDevice_t;
        typedef CUmodule GPUModule_t;
        typedef CUresult GPUError_t;
        typedef CUstream GPUStream_t;
        typedef CUcontext GPUContext_t;

        typedef unsigned long long int GPUFunctionPtr_t;

        bool GPUStreamEmpty(GPUStream_t s);

        class GPUConfig
        {
        public:
            static GPUTPCAllocatedMap_t get_available_TPCs();
            static int get_number_of_available_TPCs();
            static void allocate_TPCs(GPUTPCAllocatedMap_t sms_to_be_allocated);
            static void release_TPCs_in_map(GPUTPCAllocatedMap_t TPCs_to_be_released);
            static Status try_allocate_TPCs(int num_required_TPCs, GPUTPCAllocatedMap_t &TPCs_return_allocated);
            static Status try_allocate_TPCs_best_effort(int num_required_TPCs_required, 
                                                        int num_required_TPCs_limited, 
                                                        GPUTPCAllocatedMap_t &TPCs_return_allocated);
            static Status set_sm_mask(const GPUTPCAllocatedMap_t &TPCs_return_allocated);
            static uint32_t get_num_TPCs();
            static GPUContext_t get_gpu_ctx();
            static std::string get_gpu_device_name();
            static DeviceType get_gpu_device_type();
            static Status get_kernel_address(const char *name, GPUModule_t mod, GPUFunctionPtr_t &ret);

            struct KernelResource
            {
                int shared_memory;
                int vgprs;
                int sgprs;
                int stack_size;
            };

            static void GPUConfigInit(int device_id);

            static KernelResource max_resource(const KernelResource &kr1, const KernelResource &kr2);

            static Status get_kernel_resource(GPUFunction_t func, KernelResource &ret);

            static int calculate_occupancy(const KernelResource &resource, dim3 thread_idx);

            static uint32_t num_TPCs;

            static GPUDevice_t device;
            static GPUContext_t gpu_ctx;

            static GPUTPCAllocatedMap_t gpu_TPC_allocated_map;
            static std::mutex gpu_TPC_allocated_map_mtx;
        };

        CUresult cuResetWavefronts();

    } // namespace executor
} // namespace missilebase
