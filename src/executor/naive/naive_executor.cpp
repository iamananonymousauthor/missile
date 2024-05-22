#include <cassert>
#include <vector>
#include <queue>

#include "naive_executor.h"
#include "../../cuda_implementation/cuda_impl.h"
#include "../../server/base_scheduler.h"

namespace missilebase {
namespace executor {

NaiveExecutor::NaiveExecutor() {
    is_preempted = false;
}

NaiveExecutor::~NaiveExecutor() {}

Status NaiveExecutor::load_param_from_host_to_device() {
    if(params.get() == nullptr) {
        LOG(ERROR) << "NULL model params pointer is invalid";
        RETURN_STATUS(Status::Fail);
    }
    for (size_t i = 0; i < model->storage.size(); i++) {
        StorageInfo& storage_info = this->model->storage[i];
        if (params->find(storage_info.name) == params->end())
            continue;
        auto &array = params->at(storage_info.name);
        GPU_RETURN_STATUS(GPUMemcpyHtoD(
                (GPUDevicePtr_t)storage[i], (void*)array.h_ptr,
                array.size * sizeof(float)));
    }
    return Status::Succ;
}

Status NaiveExecutor::load_model_from_GPU_module(const char* json_file_path, 
                                                 GPUModule_t module, 
                                                 bool _is_real_time, 
                                                 int _qid) {
    Status ret = init_executor_base(json_file_path, module, false);
    if (ret != Status::Succ) return ret;
    return init_executor(json_file_path, module, _is_real_time, _qid);
}

Status NaiveExecutor::init_executor(const char* json_file_path, GPUModule_t module, bool _is_real_time, int _qid) {
    is_real_time = _is_real_time;
    qid = _qid;
    size_t num_op_calls = model->ops.size();
    kernel_args.resize(num_op_calls);
    //storage.resize(this->model->storage.size());
    raw_args.resize(num_op_calls);

    /*for (size_t i = 0; i < storage.size(); i++) {
        StorageInfo &storage_info = this->model->storage[i];
        GPU_RETURN_STATUS(GPUMalloc(
                (GPUDevicePtr_t *) &storage[i],
                storage_info.size * sizeof(float)));
    }*/

    bool need_load_kernels = true; // TODO: move to class config

    //load_param_from_host_to_device();

    // 1. fullfil the trans_args, which will be used to launch transformed kernels
    for (size_t i = 0; i < num_op_calls; i++) {
        kernel_args[i].resize(model->ops[i].kernels.size());
        raw_args[i] = std::vector<std::vector<void*>>(model->ops[i].kernels.size());
        for(size_t j = 0; j < model->ops[i].kernels.size(); j++) {
            KernelArg &kernel_arg = kernel_args[i][j];

            raw_args[i][j] = std::vector<void*>(model->ops[i].kernels[j].args.size());

            std::string &kernel_name = model->ops[i].kernels[j].name;

            uint32_t *launch_params = model->ops[i].kernels[j].launch_params;
            kernel_arg.task_dim = dim3(launch_params[0], launch_params[1], launch_params[2]);
            kernel_arg.thread_dim = dim3(launch_params[3], launch_params[4], launch_params[5]);
            kernel_arg.block_num = launch_params[0] * launch_params[1] * launch_params[2];
            kernel_arg.block_offset = 0;
            kernel_arg.cu_lower = 0;
            kernel_arg.cu_upper = GPUConfig::get_num_TPCs();

            if (need_load_kernels) {
                RETURN_STATUS(
                        GPUConfig::get_kernel_address(
                                kernel_name.c_str(), module, kernel_arg.funcion_pointer
                        )
                );
                kernel_arg.kernel = kernels_by_name[kernel_name];
                //GPU_RETURN_STATUS(GPUCtxSetCurrent(GPUConfig::get_gpu_ctx()));
                RETURN_STATUS(
                        GPUConfig::get_kernel_resource(
                                kernel_arg.kernel,
                                kernel_arg.resource
                        )
                );
            }
        }
    }
    LOG(WARNING) << "DEBUG: setup eviction flag";
    d_ptr_to_eviction_flag = NULL;
    return Status::Succ;
}

void NaiveExecutor::setup_kernel_latency_recorder() {
    record_latency_per_kernel = true;
    kernel_latency_csv_file_path = model_file_prefix + std::string(".kernel_latency.csv");
    kernel_latency_fp = fopen(kernel_latency_csv_file_path.c_str(), "w");
    ASSERT_MSG(kernel_latency_fp != NULL, "Invalid kernel latency fp: " << kernel_latency_csv_file_path);
}

    Status NaiveExecutor::set_input(
            const std::string& key, const std::vector<float>& value) {
        return set_input(key, (void*)value.data(), value.size() * sizeof(float));
    }

    Status NaiveExecutor::set_input(const std::string& key, const void* value, size_t len) {
        size_t input_storage_idx;
        if (find_storage_idx(key, input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        if (len < storage_size) RETURN_STATUS(Status::OutOfRange);
        GPU_RETURN_STATUS(GPUMemcpyHtoD(
                (GPUDevicePtr_t)this->storage[input_storage_idx], (void*)value,
                storage_size)
        );
        //RETURN_STATUS(WaitGPUMemcpyRequestComplete());
        return Status::Succ;
    }

    Status NaiveExecutor::set_input(int idx, const void* value, size_t len) {
        if (idx >= model->storage.size()) RETURN_STATUS(Status::OutOfRange);
        StorageInfo& storage_info = this->model->storage[idx];
        size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        if (len < storage_size) RETURN_STATUS(Status::OutOfRange);
        GPU_RETURN_STATUS(GPUMemcpyHtoD(
                (GPUDevicePtr_t)this->storage[idx], (void*)value,
                storage_size)
        );
        //RETURN_STATUS(WaitGPUMemcpyRequestComplete());
        return Status::Succ;
    }

    Status NaiveExecutor::get_output(std::vector<float>& out) {
        size_t input_storage_idx;
        if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        if (Model::get_stype_size(storage_info.stype) != sizeof(float)) RETURN_STATUS(Status::Fail);
        out.resize(storage_info.size);
        return get_data(input_storage_idx, (void*)out.data(), storage_info.size * sizeof(float));
    }

    Status NaiveExecutor::get_output(void* out, size_t len) {
        size_t input_storage_idx;
        if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        if (len < storage_size) RETURN_STATUS(Status::Fail);
        return get_data(input_storage_idx, out, len);
    }

    Status NaiveExecutor::get_data(int idx, void* out, size_t len) {
        if (idx >= model->storage.size()) RETURN_STATUS(Status::OutOfRange);
        StorageInfo& storage_info = this->model->storage[idx];
        size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        if (len < storage_size) RETURN_STATUS(Status::Fail);
        GPU_RETURN_STATUS(GPUMemcpyDtoH( //GPUMemcpyDtoH(
                out, (GPUDevicePtr_t)this->storage[idx], storage_size
        ));
        //RETURN_STATUS(WaitGPUMemcpyRequestComplete());
        return Status::Succ;
    }

    Status NaiveExecutor::execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores) {
        for(; launch_step_offset < model->ops.size(); launch_step_offset++) {
            //auto start_time = std::chrono::system_clock::now();
            for (int kernel_idx = 0; kernel_idx < model->ops[launch_step_offset].kernels.size(); kernel_idx++) {
                KernelArg &kernel_arg = kernel_args[launch_step_offset][kernel_idx];
                for(int arg_idx = 0; arg_idx < model->ops[launch_step_offset].kernels[kernel_idx].args.size(); arg_idx++) {
                    int storage_idx = model->ops[launch_step_offset].kernels[kernel_idx].args[arg_idx];
                    StorageInfo &storage_info = this->model->storage[storage_idx];
                        GPU_RETURN_STATUS(GPUMalloc(
                                (GPUDevicePtr_t *) &storage[storage_idx],
                                storage_info.size * sizeof(float)));
                    ASSERT_MSG(launch_step_offset < raw_args.size() &&
                                kernel_idx < raw_args[launch_step_offset].size() &&
                                arg_idx < raw_args[launch_step_offset][kernel_idx].size(),
                                "Invalid arg index " << launch_step_offset << "->" << kernel_idx << "->" << arg_idx);
                    raw_args[launch_step_offset][kernel_idx][arg_idx] = (void*) (&this->storage[storage_idx]);
                }
                kernel_arg.args = (void **) raw_args[launch_step_offset][kernel_idx].data();
                RETURN_STATUS(launch_kernel(launch_step_offset, 
                                            kernel_idx, stream, TPC_request_num_cores, 
                                            TPC_limit_num_cores));
                //auto end_time = std::chrono::system_clock::now();
                for(auto& storage_idx : model->ops[launch_step_offset].kernels[kernel_idx].args) {
                    StorageInfo &storage_info = this->model->storage[storage_idx];
                    GPU_RETURN_STATUS(GPUFree((GPUDevicePtr_t) storage[storage_idx]));
                }
            }
        }
        return Status::Succ;
    }

Status NaiveExecutor::launch_kernel(int op_offset, 
                                    int kernel_offset, 
                                    GPUStream_t stream, 
                                    int TPC_request_num_cores, 
                                    int TPC_limit_num_cores) {
    std::string& func_name = this->model->ops[op_offset].kernels[kernel_offset].name;
    printf("DEBUG: Launch kernel: %s\n", func_name.c_str());
    //ASSERT_MSG(this->kernels.find(func_name) != this->kernels.end(), "Failed to find kernel name: " << func_name);
    ASSERT_MSG(op_offset < this->kernel_args.size() && kernel_offset <this->kernel_args[op_offset].size(), 
               "Invalid index: "<<op_offset<<"->"<<kernel_offset);
    GPUFunction_t func = this->kernels[op_offset][kernel_offset];
    //int num_cus = GPUConfig::get_num_TPCs()*2;
    KernelArg &kernel_arg = this->kernel_args[op_offset][kernel_offset];
    //int logical_layers = align_up(kernel_arg.block_num, num_cus) / num_cus;

 
    if(record_latency_per_kernel) {
        int repeat_num = 10;
        auto start = std::chrono::system_clock::now();
        for(int i=0; i<repeat_num; i++) {
            GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
                                                    kernel_arg.task_dim.x, kernel_arg.task_dim.y, kernel_arg.task_dim.z,
                                                    kernel_arg.thread_dim.x, kernel_arg.thread_dim.y,
                                                    kernel_arg.thread_dim.z,
                                                    0, stream, kernel_arg.args, 0)); //TODO
        }
        GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
        auto end = std::chrono::system_clock::now();
        double elapsed_time_ms = std::chrono::duration_cast<std::chrono::microseconds>
                                                            (end - start).count() / 1000.0 / repeat_num;
        fprintf(kernel_latency_fp, "%s,%lf\n", func_name.c_str(), elapsed_time_ms);
    } else {
        GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
                                                kernel_arg.task_dim.x, kernel_arg.task_dim.y, kernel_arg.task_dim.z,
                                                kernel_arg.thread_dim.x, kernel_arg.thread_dim.y, kernel_arg.thread_dim.z,
                                                0, stream, kernel_arg.args, 0)); //TODO
        GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
    }
    return Status::Succ;
}
} // namespace executor
} // namespace missilebase