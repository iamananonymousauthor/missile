#include "executor/executor_base.h"
#include <vector>
#include <glog/logging.h>

#include "../util/common.h"
#include "../server/base_scheduler.h"

namespace missilebase {
namespace executor{

    ExecutorBase::ExecutorBase() {

    }

    ExecutorBase::~ExecutorBase() {
        // TODO: free GPU memory
    }

    void ExecutorBase::set_scheduler(missilebase::server::BaseScheduler *scheduler) {
        this->scheduler.reset(scheduler);
    }

    Status ExecutorBase::load_model_from_file( const char* json_file_path, 
                                               const char* co_file_path,
                                               const bool _is_real_time,
                                               const int _qid)
    {
        //GPU_RETURN_STATUS(GPUInit(0));
        // CUcontext ctx;
        //GPUDevice_t device;
        GPU_RETURN_STATUS(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
        // GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device)); //TODO!
        GPUModule_t mod;
        GPU_RETURN_STATUS(GPUModuleLoad(&mod, co_file_path));
        model_file_prefix = split_by_delimiter(std::string(json_file_path), '.')[0];
        return this->load_model_from_GPU_module(json_file_path, mod, _is_real_time, _qid);
    }

    Status ExecutorBase::load_model_from_GPU_module(const char* json_file_path, 
                                                    GPUModule_t mod, 
                                                    bool _is_real_time, int _qid) {
        return init_executor_base(json_file_path, mod, false);
    }


    Status ExecutorBase::init_executor_base(const char* json_file_path,
                                            GPUModule_t mod,
                                            bool load_execution_plan)
    {
        base_mod = mod;

        // 1. load json model file
        model.reset(Model::from_json(json_file_path, load_execution_plan));
        if (model.get() == nullptr) {
            RETURN_STATUS(Status::NotFound);
        }

        // 2. load hip kernels
        kernels.resize(model->ops.size());
        for(int i=0; i<model->ops.size(); i++) {
            OpInfo &op_info = model->ops[i];
            kernels[i].resize(model->ops[i].kernels.size());
            for (int j=0; j<op_info.kernels.size(); j++) {
                KernelInfo &kernel_info = op_info.kernels[j];
                GPUFunction_t kernel;
                if(kernels_by_name.find(kernel_info.name) != kernels_by_name.end()) {
                    kernels[i][j] = kernels_by_name[kernel_info.name];
                    continue;
                }
                GPU_RETURN_STATUS(
                        GPUModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
                );
                kernels_by_name[kernel_info.name] = kernel;
                kernels[i][j] = kernel;
            }
        }
        //storage.resize(model->storage.size());

        LOG(INFO) << "create base model stream";
        //TODO!!!!
        //GPU_RETURN_STATUS(hipStreamCreateWithWindowSize(&s, 16));
        return Status::Succ;
    }

    Status ExecutorBase::load_param_from_file_to_host(
        const char* param_file_path) {
        LOG(INFO) << "DEBUG: ExecutorBase load_param_from_file_to_host() = " << param_file_path;
        LOG(WARNING) << "Not implemented!";
        return Status::Succ;
    }

    // TODO: For testing only
    Status ExecutorBase::load_param_zeros_to_host() {
        params = std::make_shared<ModelParam>();
        for (size_t i = 0; i < model->storage.size(); i++) {
            StorageInfo &storage_info = this->model->storage[i];
            ParamData data;
            data.size = storage_info.size;
            uint64_t padded_size_bytes = Model::get_stype_size(storage_info.stype) * storage_info.size;
            storage_info.alloced_size_bytes = align_up((uint64_t)padded_size_bytes, (uint64_t)2048llu); //TODO!!!!!
            ASSERT_GPU_ERROR(GPUHostMalloc(&data.h_ptr, storage_info.alloced_size_bytes, CU_MEMHOSTALLOC_DEVICEMAP));
            ASSERT_MSG(params->find(storage_info.name) == params->end(), "Duplicated storage name: " << storage_info.name);
            params->insert({storage_info.name, data});
        }
        return Status::Succ;
    }

    Status ExecutorBase::load_param_from_host_to_device() {
        LOG(INFO) << "ExecutorBase's load_param_from_host_to_device() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::set_input(const std::string& key, const std::vector<float>& value) {
        LOG(INFO) << "ExecutorBase's set_input() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::set_input(int idx, const void* value, size_t len) {
        LOG(INFO) << "ExecutorBase's set_input() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::set_input(const std::string& key, const void* value, size_t len) {
        LOG(INFO) << "ExecutorBase's set_input() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::get_output(std::vector<float>& out) {
        LOG(INFO) << "ExecutorBase's get_output() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::get_output(void* out, size_t len) {
        LOG(INFO) << "ExecutorBase's get_output() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::get_data(int idx, void* out, size_t len) {
        LOG(INFO) << "ExecutorBase's get_data() has not been implemented!";
        return Status::Fail;
    }

    Status ExecutorBase::get_data_size(const std::string& key, size_t &size) {
        size_t input_storage_idx;
        if (find_storage_idx(key, input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);   
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        return Status::Succ;
    }

    Status ExecutorBase::find_storage_idx(const std::string& name, size_t& idx) {
        // TODO: O(n) -> O(1)
        for (size_t i = 0; i < model->storage.size(); i++) {
            StorageInfo& storage_info = this->model->storage[i];
            if (storage_info.name == name) {
                idx = i;
                return Status::Succ;
            }
        }
        RETURN_STATUS(Status::NotFound);
        return Status::NotFound; // otherwise, the compiler thinks no return value.
    }

    size_t ExecutorBase::num_ops() const {
        return model->ops.size();
    }

    void ExecutorBase::set_stream(GPUStream_t stream) {
        s = stream;
    }


    GPUStream_t ExecutorBase::stream() const {
        return s;
    }

    Status ExecutorBase::load_profile_data_to_host(const char* csv_file_path) {
        LOG(INFO) << "Loading ncu profile data from: " << csv_file_path;
        FILE *fp = fopen(csv_file_path, "r");
        if(fp == nullptr) {
            return Status::InvalidArgument;
        }
        char kernel_name[300];
        double dram_throughput_perc, compute_sm_perc, latency_ms, dummy1, dummy2, dummy3, dummy4, 
               max_num_blocks_per_sm, sm_needed;
        while(fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf", kernel_name, &dram_throughput_perc, 
                                                                  &dummy1, &dummy2, &dummy3, &dummy4, &compute_sm_perc, 
                                                                  &latency_ms, &max_num_blocks_per_sm, &sm_needed) != EOF) {
            map_kernel_name_to_kernel_profile[std::string(kernel_name)] = KernelProfile(dram_throughput_perc, 
                                                                                        compute_sm_perc, 
                                                                                        latency_ms, 
                                                                                        (int)max_num_blocks_per_sm, 
                                                                                        sm_needed);
        }
        return Status::Succ;
    }

    Status ExecutorBase::execute(int &create_step_offset, int &swap_in_step_offset, int &swap_out_step_offset,
                                int &launch_step_offset, int &reclaim_step_offset,  GPUStream_t stream, 
                                int TPC_request_num_cores, int TPC_limit_num_cores) {
        this->create_step_offset = create_step_offset;
        this->swap_in_step_offset = swap_in_step_offset;
        this->swap_out_step_offset = swap_out_step_offset;
        this->launch_step_offset = launch_step_offset;
        this->reclaim_step_offset = reclaim_step_offset;
        Status ret = execute_to(stream, TPC_request_num_cores, TPC_limit_num_cores);
        create_step_offset = this->create_step_offset;
        swap_in_step_offset = this->swap_in_step_offset;
        swap_out_step_offset = this->swap_out_step_offset;
        launch_step_offset = this->launch_step_offset;
        reclaim_step_offset = this->reclaim_step_offset;
        return ret;
    }

    Status ExecutorBase::execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores) {
        return Status::Fail;
    }

    Status ExecutorBase::launch_kernel(int op_offset, 
                                       int kernel_offset, 
                                       GPUStream_t stream, 
                                       int TPC_request_num_cores, 
                                       int TPC_limit_num_cores) {
        return Status::Fail;
    }

    Status ExecutorBase::TryToEvict(bool set_eviction_flag) {
        if(is_real_time) {
            RETURN_STATUS(Status::InvalidArgument);
        }
        is_preempted = true;
        if(set_eviction_flag) {
            int eviction_flag = 1;
            ASSERT_GPU_ERROR(GPUMemset((GPUDevicePtr_t) d_ptr_to_eviction_flag, 0xf, sizeof(int)));
        }
        return Status::Succ;
    }

    Status ExecutorBase::ClearEvictionFlag(bool set_eviction_flag) {
        if(is_real_time) {
            RETURN_STATUS(Status::InvalidArgument);
        }
        is_preempted = false;
        if(set_eviction_flag) {
            int eviction_flag = 0;
            ASSERT_GPU_ERROR(GPUMemset((GPUDevicePtr_t) d_ptr_to_eviction_flag, 0, sizeof(int)));
        }
        return Status::Succ;
    }

    void ExecutorBase::set_TPCs_allocated(const GPUTPCAllocatedMap_t &_TPCs_allocated) {
        TPCs_allocated = _TPCs_allocated;
    }

    int ExecutorBase::get_used_num_TPCs() {
        return TPCs_allocated.count();
    }

} // namespace executor
} // namespace missilebase