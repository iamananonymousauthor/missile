#pragma once
#include <vector>
#include "util/model.h"
#include "util/common.h"
#include "util/kernel_profile.h"
#include "../server/step_graph.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#endif

namespace missilebase {
    namespace server {
        class BaseScheduler;
    }
}

namespace missilebase {
namespace executor {

class ExecutorBase {
public:
    ExecutorBase();
    virtual ~ExecutorBase();

    void set_scheduler(missilebase::server::BaseScheduler *scheduler);

    virtual Status load_param_from_host_to_device();

    Status load_model_from_file(const char* json_file_path, 
                                const char* co_file_path, 
                                const bool _is_real_time, 
                                const int _qid);

    virtual Status load_model_from_GPU_module(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid);

    Status load_param_from_file_to_host(const char* param_file_path);

    virtual Status load_param_zeros_to_host();

    virtual Status set_input(const std::string& key, const std::vector<float>& value);

    virtual Status set_input(int idx, const void* value, size_t len);

    virtual Status set_input(const std::string& key, const void* value, size_t len);

    virtual Status get_output(std::vector<float>& out);

    virtual Status get_output(void* out, size_t len);

    virtual Status get_data(int idx, void* out, size_t len);

    Status get_data_size(const std::string& key, size_t &size);

    Status execute(int &create_step_offset, int &swap_in_step_offset, int &swap_out_step_offset,
                   int &launch_step_offset, int &reclaim_step_offset,
                   GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores);

    virtual Status execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores);

    virtual Status launch_kernel(int op_offset, 
                                 int kernel_offset, 
                                 GPUStream_t stream, 
                                 int TPC_request_num_cores, 
                                 int TPC_limit_num_cores);

    size_t num_ops() const;

    void set_stream(GPUStream_t stream);

    GPUStream_t stream() const;

    Status TryToEvict(bool set_eviction_flag = true);

    Status ClearEvictionFlag(bool set_eviction_flag = true);

    int get_used_num_TPCs();

    void set_TPCs_allocated(const GPUTPCAllocatedMap_t &TPCs_allocated);

    std::shared_ptr<Model> model;

public:
    class KernelArg {
    public:
        GPUFunction_t kernel;
        GPUFunctionPtr_t funcion_pointer;
        dim3 task_dim;
        dim3 thread_dim;
        // GPUDeviceptr_t task_slots;
        int block_num;
        int block_offset;
        int cu_lower;
        int cu_upper;
        void** args;

        int min_occupancy; // This is the minimal required occupancy for real-time task.
        GPUConfig::KernelResource resource;
        missilebase::utils::KernelProfile profile;
    };

    std::shared_ptr<StepGraph> step_graph;

protected:
    Status init_executor_base(const char* json_file_path, GPUModule_t module, bool load_execution_plan);

    Status find_storage_idx(const std::string& name, size_t &idx);

protected:
    std::string model_file_prefix;

    int used_num_TPCs; // the number of TPCs that is currently used. Only valid when state = Executing.
    GPUTPCAllocatedMap_t TPCs_allocated; // Only valid for real-time tasks
    int nice;

    bool need_execution_graph;
    bool is_preempted;
    bool is_real_time;
    int qid;
    std::shared_ptr<ModelParam> params;
    std::atomic<GPUDevicePtr_t> storage[10240];
    // on_device[i]=true indicates that the device space has been created for the i-th tensor
    std::atomic<bool> allocated_on_device[10240]; 
    std::map<std::string, GPUFunction_t> kernels_by_name;
    std::vector<std::vector<GPUFunction_t>> kernels;
    std::vector<std::vector<KernelArg>> kernel_args;
    std::vector<std::vector<std::vector<void*>>> raw_args;
    std::map<std::string, missilebase::utils::KernelProfile> map_kernel_name_to_kernel_profile;

    std::shared_ptr<StepGraph> cold_start_step_graph, warm_start_step_graph;

    GPUModule_t base_mod;
    GPUStream_t s;

    std::shared_ptr<missilebase::server::BaseScheduler> scheduler;
    int* d_ptr_to_eviction_flag;
    int create_step_offset, swap_in_step_offset, swap_out_step_offset, launch_step_offset, reclaim_step_offset;

    Status load_profile_data_to_host(const char* csv_file_path);
};


} // namespace executor
} // namespace missilebase