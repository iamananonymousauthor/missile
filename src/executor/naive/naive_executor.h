#pragma once 
#include <vector>
#include <queue>
#include <condition_variable>

#include "../executor_base.h"
#include "../cuda_implementation/cuda_impl.h"
#include "util/threadsafe_queue.h"


namespace missilebase {
namespace executor {
class NaiveExecutor : public ExecutorBase {
public:
    NaiveExecutor();
    ~NaiveExecutor();

    Status load_param_from_host_to_device();

    Status load_model_from_GPU_module(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid);

    Status set_input(const std::string& key, const std::vector<float>& value);

    Status set_input(int idx, const void* value, size_t len);

    Status set_input(const std::string& key, const void* value, size_t len);

    Status get_output(std::vector<float>& out);

    Status get_output(void* out, size_t len);

    Status get_data(int idx, void* out, size_t len);

    Status execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores);

    virtual Status launch_kernel(int op_offset, 
                                 int kernel_offset, 
                                 GPUStream_t stream, 
                                 int TPC_request_num_cores, 
                                 int TPC_limit_num_cores) override;

    void setup_kernel_latency_recorder();

protected:
    Status init_executor(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid);

protected:
    void** func_args_base_ptr_host;

    bool record_latency_per_kernel;
    std::string kernel_latency_csv_file_path;
    FILE *kernel_latency_fp;

    virtual std::string MISSILE_PROXY_KERNEL_PREFIX() const {
        return "merge_framework_";
    } 
    virtual std::string MISSILE_PROXY_KERNEL_NOSTACK_PREFIX() const {
        return "merge_framework_nostack_";
    }

};

} // namespace executor
} // namespace missilebase