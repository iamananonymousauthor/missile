#include <cassert>
#include <vector>
#include <queue>

#include "spatial_multiplexing_executor.h"
#include "../../cuda_implementation/cuda_impl.h"
#include "../../server/base_scheduler.h"
#include "../../server/spatial_multiplexing/spatial_multiplexing_scheduler.h"
#include "../../server/step_graph.h"
#include "../../util/kernel_profile.h"

namespace missilebase {
namespace executor {

using namespace missilebase::utils;

SpatialMultiplexingExecutor::SpatialMultiplexingExecutor() {
    is_preempted = false;
}

SpatialMultiplexingExecutor::~SpatialMultiplexingExecutor() {}

Status SpatialMultiplexingExecutor::load_param_from_host_to_device() {
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

Status SpatialMultiplexingExecutor::load_model_from_GPU_module(const char* json_file_path, 
                                                               GPUModule_t module, 
                                                               bool _is_real_time, 
                                                               int _qid) {
    Status ret = init_executor_base(json_file_path, module, true);
    if (ret != Status::Succ) return ret;
    return init_executor(json_file_path, module, _is_real_time, _qid);
}

Status SpatialMultiplexingExecutor::init_executor(const char* json_file_path, 
                                                  GPUModule_t module, 
                                                  bool _is_real_time, 
                                                  int _qid) {
    is_cold_start = true;
    is_real_time = _is_real_time;
    qid = _qid;
    size_t num_op_calls = model->ops.size();
    kernel_args.resize(num_op_calls);
    //storage.resize(this->model->storage.size());
    raw_args.resize(num_op_calls);
    //storage.resize(model->storage.size());
    for(int i=0; i<model->storage.size(); i++) {
        storage[i] = NULL;
    }

    RETURN_STATUS(load_profile_data_to_host((model_file_prefix+std::string(".ncuprofile.csv")).c_str()));

    cold_start_step_graph = std::make_shared<StepGraph>(this->model, true);
    warm_start_step_graph = std::make_shared<StepGraph>(this->model, false);
    step_graph = cold_start_step_graph;

    bool need_load_kernels = true; // TODO: move to class config

    //load_param_from_host_to_device();

    // 1. fullfil the trans_args, which will be used to launch transformed kernels
    for (size_t i = 0; i < num_op_calls; i++) {
        kernel_args[i].resize(model->ops[i].kernels.size());
        raw_args[i] = std::vector<std::vector<void*>>(model->ops[i].kernels.size());
        for(size_t j = 0; j < model->ops[i].kernels.size(); j++) {
            KernelArg &kernel_arg = kernel_args[i][j];

            /*raw_args[i][j] = std::vector<void*>(model->ops[i].kernels[j].args.size());
            for (int arg_idx = 0; arg_idx < model->ops[i].kernels[j].args.size(); arg_idx++) {
                uint32_t storage_idx = model->ops[i].kernels[j].args[arg_idx];
                raw_args[i][j][arg_idx] = (void*) (&this->storage[storage_idx]);
            }
            kernel_arg.args = (void **) raw_args[i][j].data();*/

            std::string &kernel_name = model->ops[i].kernels[j].name;

            uint32_t *launch_params = model->ops[i].kernels[j].launch_params;
            kernel_arg.task_dim = dim3(launch_params[0], launch_params[1], launch_params[2]);
            kernel_arg.thread_dim = dim3(launch_params[3], launch_params[4], launch_params[5]);
            kernel_arg.block_num = launch_params[0] * launch_params[1] * launch_params[2];
            kernel_arg.block_offset = 0;
            kernel_arg.cu_lower = 0;
            kernel_arg.cu_upper = GPUConfig::get_num_TPCs();
            kernel_arg.profile = map_kernel_name_to_kernel_profile[kernel_name];
        }
    }

    if(is_real_time) {
        d_ptr_to_eviction_flag = NULL;
    } else {
        GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t*)&d_ptr_to_eviction_flag, sizeof(int)));
    }
    ASSERT_GPU_ERROR(GPUStreamCreate(&stream_swap_in));
    ASSERT_GPU_ERROR(GPUStreamCreate(&stream_swap_out));
    return Status::Succ;
}

    Status SpatialMultiplexingExecutor::set_input(
            const std::string& key, const std::vector<float>& value) {
        return set_input(key, (void*)value.data(), value.size() * sizeof(float));
    }

    Status SpatialMultiplexingExecutor::set_input(const std::string& key, const void* value, size_t len) {
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

    Status SpatialMultiplexingExecutor::set_input(int idx, const void* value, size_t len) {
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

    Status SpatialMultiplexingExecutor::get_output(std::vector<float>& out) {
        size_t input_storage_idx;
        if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        if (Model::get_stype_size(storage_info.stype) != sizeof(float)) RETURN_STATUS(Status::Fail);
        out.resize(storage_info.size);
        return get_data(input_storage_idx, (void*)out.data(), storage_info.size * sizeof(float));
    }

    Status SpatialMultiplexingExecutor::get_output(void* out, size_t len) {
        size_t input_storage_idx;
        if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
        StorageInfo& storage_info = this->model->storage[input_storage_idx];
        size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
        if (len < storage_size) RETURN_STATUS(Status::Fail);
        return get_data(input_storage_idx, out, len);
    }

    Status SpatialMultiplexingExecutor::get_data(int idx, void* out, size_t len) {
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

    Status SpatialMultiplexingExecutor::execute_to_create(int &create_step_offset) {
        for(; create_step_offset < step_graph->create_step_node_list.size(); create_step_offset++) {
            ASSERT_MSG(step_graph->create_step_node_list[create_step_offset]->wait_for_prerequisites() == Status::Succ,
                       "Timeout at create step #" << create_step_offset);
            // Do cuMemAlloc
            int storage_idx = step_graph->create_step_node_list[create_step_offset]->target_idx;
            assert(storage_idx >= 0);
            assert(storage_idx < this->model->storage.size());
            ASSERT_MSG(this->storage[storage_idx] == NULL, "Create step ["<< create_step_offset <<
                        "] points at a valid pointer (storage idx [" << storage_idx << "])");
            StorageInfo& storage_info = this->model->storage[storage_idx];
            GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t *) &this->storage[storage_idx],
                                        Model::get_stype_size(storage_info.stype) * storage_info.size));
            step_graph->create_step_node_list[create_step_offset]->mark_finished();
        }
        return Status::Succ;
    }

    Status SpatialMultiplexingExecutor::execute_to_swap_in(int &swap_in_step_offset, GPUStream_t stream) {
        for(; swap_in_step_offset < step_graph->swap_in_step_node_list.size(); swap_in_step_offset++) {
            ASSERT_MSG(step_graph->swap_in_step_node_list[swap_in_step_offset]->wait_for_prerequisites() == Status::Succ,
                       "Timeout at Swap in step #" << swap_in_step_offset);
            // Do memcpy H2D
            int storage_idx = step_graph->swap_in_step_node_list[swap_in_step_offset]->target_idx;
            assert(storage_idx >= 0);
            assert(storage_idx < this->model->storage.size());
            ASSERT_MSG(this->storage[storage_idx] != NULL, "Swap in step ["<< swap_in_step_offset <<
                    "] points at a NULL pointer (storage idx [" << storage_idx << "])");
            StorageInfo& storage_info = this->model->storage[storage_idx];
            if (params->find(storage_info.name) == params->end())
                continue;
            auto &array = params->at(storage_info.name);
            GPU_RETURN_STATUS(GPUMemcpyHtoDAsync(
                    (GPUDevicePtr_t)this->storage[storage_idx], (void*)array.h_ptr,
                    Model::get_stype_size(storage_info.stype) * storage_info.size, stream));
            step_graph->swap_in_step_node_list[swap_in_step_offset]->mark_finished();
        }
    }

    Status SpatialMultiplexingExecutor::execute_to_swap_out(int &swap_out_step_offset, GPUStream_t stream) {
        for(; swap_out_step_offset < step_graph->swap_out_step_node_list.size(); swap_out_step_offset++) {
                        ASSERT_MSG(step_graph->swap_out_step_node_list[swap_out_step_offset]->wait_for_prerequisites() == 
                                   Status::Succ,
                                   "Timeout at swap out step #" << swap_out_step_offset);
            // Do memcpy D2H
            int storage_idx = step_graph->swap_out_step_node_list[swap_out_step_offset]->target_idx;
            assert(storage_idx >= 0);
            assert(storage_idx < this->model->storage.size());
            ASSERT_MSG(this->storage[storage_idx] != NULL, "Swap out step ["<< swap_out_step_offset <<
                        "] points at a NULL pointer (storage idx [" << storage_idx << "])");
            StorageInfo& storage_info = this->model->storage[storage_idx];
            if (params->find(storage_info.name) == params->end())
                continue;
            auto &array = params->at(storage_info.name);
            GPU_RETURN_STATUS(GPUMemcpyDtoHAsync(
                    (void*)array.h_ptr, (GPUDevicePtr_t)this->storage[storage_idx],
                    Model::get_stype_size(storage_info.stype) * storage_info.size, stream));
            GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
            step_graph->swap_out_step_node_list[swap_out_step_offset]->mark_finished();
        }
        return Status::Succ;
    }

    Status SpatialMultiplexingExecutor::execute_to_launch(int &launch_step_offset, 
                                                          GPUStream_t stream, 
                                                          int TPC_request_num_cores, 
                                                          int TPC_limit_num_cores) {
        for(; launch_step_offset < step_graph->launch_step_node_list.size(); launch_step_offset++) {
            //auto start_time = std::chrono::system_clock::now();
            if (is_preempted) {
                return Status::Succ;
            }
            int op_idx = step_graph->launch_step_node_list[launch_step_offset]->target_idx;

            ASSERT_MSG(step_graph->launch_step_node_list[launch_step_offset]->wait_for_prerequisites() == Status::Succ,
                       "Timeout at launch step #" << launch_step_offset);

            GPU_RETURN_STATUS(GPUStreamSynchronize(stream_swap_in));
            //std::vector<std::vector<void*>> passed_ptr_to_args_list(model->ops[op_idx].kernels.size());
            for(int kernel_idx = 0; kernel_idx < model->ops[op_idx].kernels.size(); kernel_idx++) {
                KernelArg &kernel_arg = kernel_args[op_idx][kernel_idx];

                raw_args[op_idx][kernel_idx] = std::vector<void*>(model->ops[op_idx].kernels[kernel_idx].args.size());
                for (int arg_idx = 0; arg_idx < model->ops[op_idx].kernels[kernel_idx].args.size(); arg_idx++) {
                    uint32_t storage_idx = model->ops[op_idx].kernels[kernel_idx].args[arg_idx];
                    ASSERT_MSG(this->storage[storage_idx] != NULL, "Launch step ["
                               << launch_step_offset <<"] -> Kernel [" << kernel_idx
                               << "] -> Arg [" << arg_idx 
                               << "] points at a NULL pointer (storage idx [" << storage_idx << "])");
                    raw_args[op_idx][kernel_idx][arg_idx] = (void*) (&this->storage[storage_idx]);
                }
                kernel_arg.args = (void **) raw_args[op_idx][kernel_idx].data();
            }
            for (int kernel_idx = 0; kernel_idx < model->ops[op_idx].kernels.size(); kernel_idx++) {
                std::string& func_name = this->model->ops[op_idx].kernels[kernel_idx].name;
                if(this->scheduler->mode == server::MultiTaskPolicy) {
                    RETURN_STATUS(launch_kernel(op_idx, kernel_idx, stream, TPC_request_num_cores,
                                                                          TPC_limit_num_cores));
                } else if (this->scheduler->mode == server::OrionPolicy) {
                    RETURN_STATUS(this->scheduler->request_launch_kernel(this->qid, this->is_real_time, op_idx,
                                                                          kernel_idx, TPC_request_num_cores,
                                                                          TPC_limit_num_cores,
                                                                          (double)launch_step_offset / (double)step_graph->launch_step_node_list.size(),
                                                                          kernel_args[op_idx][kernel_idx].profile.duration_ms));
                    if(is_real_time) {
                        this->scheduler->rt_task_gpu_launch_kernel_queues[this->qid]->wait_until_empty();
                    } else {
                        this->scheduler->be_task_gpu_launch_kernel_queues[this->qid]->wait_until_empty();
                    }
                } else {
                    ASSERT_MSG(false, "Invalid Scheduler Mode: " << this->scheduler->mode);
                }
            }
            step_graph->launch_step_node_list[launch_step_offset]->mark_finished();
        }
    }

    Status SpatialMultiplexingExecutor::execute_to_reclaim(int &reclaim_step_offset) {
        for(; reclaim_step_offset < step_graph->reclaim_step_node_list.size(); reclaim_step_offset++) {
            int storage_idx = step_graph->reclaim_step_node_list[reclaim_step_offset]->target_idx;
            ASSERT_MSG(step_graph->reclaim_step_node_list[reclaim_step_offset]->wait_for_prerequisites() == Status::Succ,
                       "Timeout at reclaim step #" << reclaim_step_offset);
            // Do memcpy D2H
            assert(storage_idx >= 0);
            assert(storage_idx < this->model->storage.size());
            StorageInfo& storage_info = this->model->storage[storage_idx];
            if (params->find(storage_info.name) == params->end())
                continue;
            auto &array = params->at(storage_info.name);
            ASSERT_MSG(this->storage[storage_idx] != NULL, "Reclaim step ["<< reclaim_step_offset <<
                    "] points at a NULL pointer (storage idx [" << storage_idx << "])");
            GPU_RETURN_STATUS(GPUFree((GPUDevicePtr_t)this->storage[storage_idx]));
            this->storage[storage_idx] = NULL;
            //LOG(INFO) << "DEBUG: step_graph->reclaim_step_node_list[" << reclaim_step_offset  << "]: finished.";
            step_graph->reclaim_step_node_list[reclaim_step_offset]->mark_finished();
        }
        return Status::Succ;
    }

    Status SpatialMultiplexingExecutor::execute_to(GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores) {
        if(is_preempted) {
            return Status::Succ;
        }
        if(create_step_offset == 0 && 
           swap_in_step_offset == 0 && 
           swap_out_step_offset == 0 && 
           launch_step_offset == 0 && 
           reclaim_step_offset == 0) {
            if(is_cold_start) {
                this->step_graph = this->cold_start_step_graph;
                need_execution_graph = true;
            } else {
                this->step_graph = this->warm_start_step_graph;
                if(this->warm_start_step_graph->swap_in_step_node_list.size() == 0) {
                    need_execution_graph = false;
                }
            }
            this->step_graph->reset();
            time_elapsed_launch_kernel_ms = 0;
        }
        if(need_execution_graph) {
            std::unique_ptr <std::thread> create_thread = std::make_unique<std::thread>();
            std::unique_ptr <std::thread> swap_in_thread = std::make_unique<std::thread>();
            std::unique_ptr <std::thread> swap_out_thread = std::make_unique<std::thread>();
            std::unique_ptr <std::thread> launch_thread = std::make_unique<std::thread>();
            std::unique_ptr <std::thread> reclaim_thread = std::make_unique<std::thread>();

            create_thread.reset(new std::thread([this]() {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                execute_to_create(create_step_offset);
            }));

            swap_in_thread.reset(new std::thread([this](GPUStream_t stream) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                execute_to_swap_in(swap_in_step_offset, stream_swap_in);
            }, stream));

            swap_out_thread.reset(new std::thread([this](GPUStream_t stream) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                execute_to_swap_out(swap_out_step_offset, stream_swap_out);
            }, stream));

            launch_thread.reset(
                    new std::thread([this](GPUStream_t stream, int TPC_request_num_cores, int TPC_limit_num_cores) {
                        ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                        execute_to_launch(launch_step_offset, stream, TPC_request_num_cores, TPC_limit_num_cores);
                    }, stream, TPC_request_num_cores, TPC_limit_num_cores));

            reclaim_thread.reset(new std::thread([this]() {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                execute_to_reclaim(reclaim_step_offset);
            }));

            create_thread->join();
            swap_in_thread->join();
            swap_out_thread->join();
            launch_thread->join();
            reclaim_thread->join();
        } else {
            struct timespec ns_sleep;
            ns_sleep.tv_nsec = 1000;
            for(; launch_step_offset < this->step_graph->launch_step_node_list.size(); launch_step_offset++) {
                while (is_preempted) {
                    nanosleep(&ns_sleep, 0);
                }
                int op_idx = step_graph->launch_step_node_list[launch_step_offset]->target_idx;
                for (int kernel_idx = 0; kernel_idx < model->ops[op_idx].kernels.size(); kernel_idx++) {
                    auto start = std::chrono::system_clock::now();
                    RETURN_STATUS(
                            this->scheduler->request_launch_kernel(this->qid, 
                                                         this->is_real_time, op_idx, 
                                                         kernel_idx, TPC_request_num_cores, TPC_limit_num_cores,
                                                         (double)launch_step_offset / (double)step_graph->launch_step_node_list.size(),
                                                         kernel_args[op_idx][kernel_idx].profile.duration_ms));
                    if(is_real_time) {
                        this->scheduler->rt_task_gpu_launch_kernel_queues[this->qid]->wait_until_empty();
                    } else {
                        this->scheduler->be_task_gpu_launch_kernel_queues[this->qid]->wait_until_empty();
                    }
                    auto end = std::chrono::system_clock::now();
                }
            }
        }

        if(step_graph->is_finished(create_step_offset, swap_in_step_offset,
                                   swap_out_step_offset, launch_step_offset,
                                   reclaim_step_offset)) {
            is_cold_start = false;
            step_graph = warm_start_step_graph;
            LOG(INFO) << (is_real_time?"RT":"BE") << " finished an iteration! time elapsed in launch kernel: " 
            << time_elapsed_launch_kernel_ms <<" ms";
        }

        return Status::Succ;
    }

    Status SpatialMultiplexingExecutor::launch_kernel(int op_offset, 
                                                      int kernel_offset, 
                                                      GPUStream_t stream, 
                                                      int TPC_request_num_cores, 
                                                      int TPC_limit_num_cores) {
        std::string& func_name = this->model->ops[op_offset].kernels[kernel_offset].name;
        GPUFunction_t func = this->kernels[op_offset][kernel_offset];
        int num_cus = GPUConfig::get_num_TPCs()*2;

        KernelArg &kernel_arg = this->kernel_args[op_offset][kernel_offset];
        int logical_layers = align_up(kernel_arg.block_num, num_cus) / num_cus;

        /*if(!is_real_time) {
            while (GPUConfig::try_allocate_TPCs(TPC_request_num_cores, TPCs_allocated) !=
                    Status::Succ) {
                usleep(100);
            }
        }*/
        auto start = std::chrono::system_clock::now();
        if(this->scheduler->mode == server::MultiTaskPolicy) {
            GPUConfig::set_sm_mask(TPCs_allocated);
        }
        GPU_RETURN_STATUS(GPUModuleLaunchKernel(func,
                                                kernel_arg.task_dim.x, kernel_arg.task_dim.y, kernel_arg.task_dim.z,
                                                kernel_arg.thread_dim.x, kernel_arg.thread_dim.y, kernel_arg.thread_dim.z,
                                                0, stream, kernel_arg.args, 0)); //TODO
        GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
        auto end = std::chrono::system_clock::now();
        time_elapsed_launch_kernel_ms +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;


        /*if(!is_real_time) { // If this kernel has been preempted, its TPCs will be released in the bitmap after preemption
            GPUConfig::release_TPCs_in_map(TPCs_allocated);
        }*/
        return Status::Succ;
    }

    Status SpatialMultiplexingExecutor::get_kernel_profile(int op_offset, int kernel_offset, 
                                                           KernelProfile &kernel_profile){
        std::string kernel_name = this->model->ops[op_offset].kernels[kernel_offset].name;
        kernel_profile = kernel_args[op_offset][kernel_offset].profile;
        return Status::Succ;
    }
} // namespace executor
} // namespace missilebase