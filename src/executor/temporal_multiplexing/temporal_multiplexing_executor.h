#pragma once 
#include <vector>
#include <queue>
#include <condition_variable>

#include "../executor_base.h"
#include "../cuda_implementation/cuda_impl.h"
#include "util/threadsafe_queue.h"


namespace missilebase {
namespace executor {
class TemporalMultiplexingExecutor : public ExecutorBase {
public:
    TemporalMultiplexingExecutor();
    ~TemporalMultiplexingExecutor();

    Status load_param_from_host_to_device();

    Status load_model_from_GPU_module(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid);

    Status set_input(const std::string& key, const std::vector<float>& value);

    Status set_input(int idx, const void* value, size_t len);

    Status set_input(const std::string& key, const void* value, size_t len);

    Status get_output(std::vector<float>& out);

    Status get_output(void* out, size_t len);

    Status get_data(int idx, void* out, size_t len);

    Status execute_to_create(int &create_step_offset);

    Status execute_to_launch(int &launch_step_offset, 
                             GPUStream_t stream, 
                             int TPC_request_num_cores, 
                             int TPC_limit_num_cores);

    Status execute_to_swap_in(int &swap_in_step_offset, GPUStream_t stream);

    Status execute_to_swap_out(int &swap_out_step_offset, GPUStream_t stream);

    Status execute_to_reclaim(int &reclaim_step_offset);

    Status execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores);

    virtual Status launch_kernel(int op_offset, int kernel_offset, 
                                 GPUStream_t stream, 
                                 int TPC_request_num_cores, 
                                 int TPC_limit_num_cores) override;

    bool is_cold_start;
protected:
    Status init_executor(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid);

protected:
    void** func_args_base_ptr_host;
};

} // namespace executor
} // namespace missilebase