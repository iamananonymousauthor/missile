#include "cuda_impl.h"
#include "util/common.h"

#include <cmath>
#include <assert.h>
#include <glog/logging.h>

#include "libsmctrl.h"

namespace missilebase {
namespace executor {

uint32_t GPUConfig::num_TPCs;

GPUDevice_t GPUConfig::device;
GPUContext_t GPUConfig::gpu_ctx;
std::string device_name;
DeviceType device_type;

GPUTPCAllocatedMap_t GPUConfig::gpu_TPC_allocated_map;
std::mutex GPUConfig::gpu_TPC_allocated_map_mtx;

void GPUConfig::GPUConfigInit(int device_id) {
    ASSERT_GPU_ERROR(cuInit(device_id));
    ASSERT_GPU_ERROR(cuDeviceGet(&device, 0));
    ASSERT_GPU_ERROR(cuCtxCreate(&gpu_ctx, 0, device));

    ASSERT_GPU_ERROR(cuDeviceGetAttribute((int*)&num_TPCs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    num_TPCs /= 2;

    int device_count;
    ASSERT_GPU_ERROR(cuDeviceGetCount(&device_count));

    char device_name_char[256];
    ASSERT_GPU_ERROR(cuDeviceGetName(device_name_char, 256, device));
    device_name = std::string(device_name_char);
    device_type = get_cuda_device_type(device_name);
    LOG(INFO) << "Current device: CUDA #" << device_id << "/" << device_count << " (" << device_name << "), # of TPCs: " << num_TPCs << std::endl;
}

uint32_t GPUConfig::get_num_TPCs() {
    // TODO: dynamic load CU nums
    return num_TPCs;
}

GPUContext_t GPUConfig::get_gpu_ctx() {
    return gpu_ctx;
}

std::string GPUConfig::get_gpu_device_name() {
    return device_name;
}

DeviceType GPUConfig::get_gpu_device_type() {
    return device_type;
}

int GPUConfig::get_number_of_available_TPCs() {
    std::unique_lock<std::mutex> lock(gpu_TPC_allocated_map_mtx);
    return num_TPCs - gpu_TPC_allocated_map.count();
}

GPUTPCAllocatedMap_t GPUConfig::get_available_TPCs() {
    std::unique_lock<std::mutex> lock(gpu_TPC_allocated_map_mtx);
    return gpu_TPC_allocated_map.flip(); //TODO
}

void GPUConfig::release_TPCs_in_map(GPUTPCAllocatedMap_t TPCs_to_be_released) {
    std::unique_lock<std::mutex> lock(gpu_TPC_allocated_map_mtx);
    assert(TPCs_to_be_released.count() == (TPCs_to_be_released & gpu_TPC_allocated_map).count());
    gpu_TPC_allocated_map &= TPCs_to_be_released.flip();
}

Status GPUConfig::try_allocate_TPCs(int num_required_TPCs, GPUTPCAllocatedMap_t &TPCs_return_allocated) {
    std::unique_lock<std::mutex> lock(gpu_TPC_allocated_map_mtx);
    if(num_TPCs - gpu_TPC_allocated_map.count()<num_required_TPCs) {
        #ifdef DEBUG_MODE
            LOG(WARNING) << "try_allocate_TPCs failed. Total = " << num_TPCs << ", available = " << num_TPCs - gpu_TPC_allocated_map.count();
        #endif
        return Status::Fail;
    }
    TPCs_return_allocated = 0; // Clear the return value first!
    for(int i=0, num_allocated_TPCs=0; i<num_TPCs && num_allocated_TPCs<num_required_TPCs; i++) {
        if(gpu_TPC_allocated_map[i] == 0) {
            gpu_TPC_allocated_map[i] = 1;
            TPCs_return_allocated[i] = 1;
            num_allocated_TPCs++;
        }
    }
    return Status::Succ;
}

Status GPUConfig::try_allocate_TPCs_best_effort(int num_required_TPCs_required, int num_required_TPCs_limited, GPUTPCAllocatedMap_t &TPCs_return_allocated) {
        std::unique_lock<std::mutex> lock(gpu_TPC_allocated_map_mtx);
        if(num_TPCs - gpu_TPC_allocated_map.count()<num_required_TPCs_required) {
            #ifdef DEBUG_MODE
                LOG(WARNING) << "try_allocate_TPCs failed. Total = " << num_TPCs << ", available = " << num_TPCs - gpu_TPC_allocated_map.count();
            #endif
            return Status::Fail;
        }
        TPCs_return_allocated = 0; // Clear the return value first!
        for(int i=0, num_allocated_TPCs=0; i<num_TPCs && num_allocated_TPCs < num_required_TPCs_limited; i++) {
            if(gpu_TPC_allocated_map[i] == 0) {
                gpu_TPC_allocated_map[i] = 1;
                TPCs_return_allocated[i] = 1;
                num_allocated_TPCs++;
            }
        }
        return Status::Succ;
}

Status GPUConfig::set_sm_mask(const GPUTPCAllocatedMap_t &TPCs_return_allocated) {
    libsmctrl_set_next_mask(~(TPCs_return_allocated.to_ulong()));
}

Status GPUConfig::get_kernel_address(const char* name, GPUModule_t mod, GPUFunctionPtr_t& ret) {
    // hipFunction_t temp;
    // GPU_RETURN_STATUS(hipModuleGetFunction(&temp, mod, name));
    // hipFunctionWGInfo_t wgInfo;
    // GPU_RETURN_STATUS(hipFuncGetWGInfo(temp, &wgInfo));
    // hipDeviceptr_t temp_buf;
    // GPU_RETURN_STATUS(hipMalloc(&temp_buf, 64));
    // int buf[24];
    // int size = 24;

    // GPU_RETURN_STATUS(hipMemcpyDtoD(temp_buf, (hipDeviceptr_t)wgInfo.baseAddress, size));
    // GPU_RETURN_STATUS(hipMemcpy(buf, temp_buf, size, hipMemcpyDeviceToHost));
    // GPU_RETURN_STATUS(hipFree(temp_buf));

    // ret = wgInfo.baseAddress + *(long long int*)(&buf[4]);
    return Status::Succ;  
}

GPUConfig::KernelResource GPUConfig::max_resource(
            const KernelResource& kr1, const KernelResource& kr2) {
    KernelResource ret;
    ret.sgprs = std::max(kr1.sgprs, kr2.sgprs);
    ret.vgprs = std::max(kr1.vgprs, kr2.vgprs);
    ret.shared_memory = std::max(kr1.shared_memory, kr2.shared_memory);
    ret.stack_size = std::max(kr1.stack_size, kr2.stack_size);
    return ret;
}

Status GPUConfig::get_kernel_resource(GPUFunction_t func, KernelResource& ret) {
    cudaFuncAttributes attr;
    GPU_RETURN_STATUS(cuFuncGetAttribute(&ret.shared_memory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
    GPU_RETURN_STATUS(cuFuncGetAttribute(&ret.vgprs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));
    GPU_RETURN_STATUS(cuFuncGetAttribute(&ret.stack_size, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func));
    return Status::Succ;
}

int GPUConfig::calculate_occupancy(const KernelResource& resource, dim3 block_dim) {
    // Input
    int registersPerThread = resource.vgprs; //32; //align_up(vgprs, 4); //TODO: Skip sgprs because NVIDIA doesn't differentiate sgpr & vgpr.
    int sharedMemoryPerBlock = resource.shared_memory; //2048; //align_up(resource.shared_memory, 256);
    int threadsPerBlock = (int)align_up<unsigned int>(block_dim.x * block_dim.y * block_dim.z, 64);
    // Configs
    int threadsPerWarp = 32; //TODO
    int threadBlocksPerMultiprocessor=16; //TODO
    int warpsPerMultiprocessor=32; //TODO
    int registerFileSize=256 * 1024 / 4; //TODO
    int warpAllocationGranularity=4; //TODO
    int registerAllocationUnitSize=256; //TODO
    int sharedMemoryPerMultiprocessor=65536; //TODO
    int sharedMemoryAllocationUnitSize=256; //TODO

    // Do calculation
    int blockWarps = ceil(threadsPerBlock / threadsPerWarp);

    int blockSharedMemory = (int)align_up<unsigned int>(sharedMemoryPerBlock, sharedMemoryAllocationUnitSize);

    int blockRegisters=(int)align_up<unsigned int>((int)align_up<unsigned int>(blockWarps, 
                                                            warpAllocationGranularity) * registersPerThread * threadsPerWarp, 
                                                            registerAllocationUnitSize);

    int threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor = int_min(
            threadBlocksPerMultiprocessor, int(floor(warpsPerMultiprocessor / blockWarps)));

    int threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor;
    if (registersPerThread > 0) {
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = floor(registerFileSize / blockRegisters);
    } else {
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = threadBlocksPerMultiprocessor;
    }

    int threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor;

    if (sharedMemoryPerBlock > 0) {
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = int(floor(sharedMemoryPerMultiprocessor / blockSharedMemory));
    } else {
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = threadBlocksPerMultiprocessor;
    }

    int activeThreadBlocksPerMultiprocessor = int_min(
            threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor,
            threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor);
    activeThreadBlocksPerMultiprocessor = int_min(
            activeThreadBlocksPerMultiprocessor,
            threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor);
    return activeThreadBlocksPerMultiprocessor;
}

bool GPUStreamEmpty(GPUStream_t s) {
    return cuStreamQuery(s) == CUDA_SUCCESS;
}

CUresult cuResetWavefronts() {
    printf("CANNOT RESET WAVE FRONTS FOR CUDA!\n");
    return CUDA_ERROR_INVALID_VALUE;
}

} // namespace executor
} // namespace missilebase