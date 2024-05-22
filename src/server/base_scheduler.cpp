#include <thread>
#include <gdrapi.h>

#include "base_scheduler.h"
#include "util/common.h"
#include "scheduler_utils.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#endif

#define ENABLE_TASK_CV 

namespace missilebase {
namespace server {
    BaseScheduler::BaseScheduler(int _gpu_id, ScheduleMode _mode, int _max_num_rt_queues, int _max_num_be_queues) :
            mode(_mode),
            model_pool(this->model_pool_capacity),
            model_pool_size(0),
            task_idx_pool(0),
            be_queues(_max_num_be_queues),
            be_queue_cnt(0),
            rt_queues(_max_num_rt_queues),
            rt_queue_cnt(0),
            rt_task_gpu_launch_kernel_queues(_max_num_rt_queues),
            be_task_gpu_launch_kernel_queues(_max_num_be_queues),
            _shutdown(false),
            preempted(false),
            wait_sync(false),
            preempt_count(0),
            preempt_latency_sum(0),
            kernel_sel_latency_sum(0),
            kernel_sel_count(0)
    {
        LOG(INFO) << "Init Scheduler......";
        cpu_core_manager.init(std::thread::hardware_concurrency());
        max_num_rt_queues = _max_num_rt_queues;
        max_num_be_queues = _max_num_be_queues;
        gpu_id = _gpu_id;
        missilebase::executor::GPUConfig::GPUConfigInit(gpu_id);
        ASSERT_GPU_ERROR(GPUStreamCreate(&execute_stream));
        ASSERT_GPU_ERROR(GPUStreamCreate(&preempt_stream));
        ASSERT_GPU_ERROR(GPUMalloc(&preempt_flag, 4)); //TODO
        ASSERT_GPU_ERROR(GPUWriteValue32Async(preempt_stream, preempt_flag, 0, 0));
        ASSERT_GPU_ERROR(GPUStreamSynchronize(preempt_stream));
        device_type = get_cuda_device_type(missilebase::executor::GPUConfig::get_gpu_device_name());
        LOG(INFO) << "Try to alloc pinned memory " << host_memory_pool_capacity_bytes << " bytes";
        ASSERT_GPU_ERROR(cuMemAllocHost(&host_memory_pool, host_memory_pool_capacity_bytes));
        gdr_g = gdr_open();
        if (!gdr_g) {
            fprintf(stderr, "gdr_open error: Is gdrdrv driver installed and loaded?\n");
            exit(EXIT_FAILURE);
        }

        if (mode == WaitPreempt) {
            be_stream_device_queue_cap = 1024;
        } else {
            be_stream_device_queue_cap = 2;
        }
        for(int i=0;i<_max_num_be_queues;i++) {
            avail_be_queue_ids.push(i);
            be_queues[i] = std::make_shared<TaskQueue>();
        }
        for(int i=0;i<_max_num_rt_queues;i++) {
            avail_rt_queue_ids.push(i);
            rt_queues[i] = std::make_shared<TaskQueue>();
        }

        for(int i=0; i<_max_num_rt_queues;i++) {
            rt_task_gpu_launch_kernel_queues[i] = std::make_unique<missilebase::utils::GPULaunchKernelQueue>();
        }
        for(int i=0; i<_max_num_be_queues;i++) {
            be_task_gpu_launch_kernel_queues[i] = std::make_unique<missilebase::utils::GPULaunchKernelQueue>();
        }
    }

    BaseScheduler::~BaseScheduler() {
    }

    Status BaseScheduler::create_task_queue(std::shared_ptr<TaskQueue>& ret, 
                                            bool rt, int TPC_request_num_cores, 
                                            int TPC_limit_num_cores, 
                                            int alloc_memory_size_kib, 
                                            int perc_pcie_bandwidth) {
        executor::GPUStream_t stream;
        GPU_RETURN_STATUS(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
        GPU_RETURN_STATUS(GPUStreamCreate(&stream));
        if (rt) {
            LOG(INFO) << "create rt stream, # TPC Cores: request = " << TPC_request_num_cores 
                      << " limit = " << TPC_limit_num_cores << std::endl;
        } else {
            LOG(INFO) << "create be stream, # TPC Cores: request = " << TPC_request_num_cores 
                      << " limit = " << TPC_limit_num_cores << std::endl;
        }
        LOG(INFO) << "DEBUG: create_task_queue start";
        ret = std::make_shared<TaskQueue>();
        ret->stream = stream;
        ret->TPC_request_num_cores = TPC_request_num_cores;
        ret->TPC_limit_num_cores = TPC_limit_num_cores;
        ret->alloc_memory_size_kib = alloc_memory_size_kib;
        ret->perc_pcie_bandwidth = perc_pcie_bandwidth;
        return Status::Succ;
    }

    Status BaseScheduler::load_model_to_host(
        const SchedulerType scheduler_type,
        const std::string& kernel_co_path,
        const std::string& json_path,
        const std::string& param_path,
        ModelID& mid,
        bool _is_real_time,
        int _qid,
        int _estimated_runtime_ms,
        int _slo_ms
    ) {
        LOG(INFO) << "DEBUG: start load_model_to_host, path = " << kernel_co_path;
        std::shared_ptr<TaskAbstractModel> model(new TaskAbstractModel(scheduler_type));
        model->executor->set_scheduler(this);
        model->is_real_time = _is_real_time;
        model->estimated_runtime_ms = _estimated_runtime_ms;
        model->slo_ms = _slo_ms;
        ASSERT_MSG(model->executor->load_model_from_file(
                json_path.c_str(),
                kernel_co_path.c_str(),
                _is_real_time,
                _qid
            ) == Status::Succ, "Invalid file path. Json path = " << json_path << ", "
            << "Kernel binary path = " << kernel_co_path);

        LOG(INFO) << "DEBUG: load_param path = " << param_path;
        model->qid = -1; // Invalid value as default
        if (param_path.size() > 0) {
            RETURN_STATUS(model->executor->load_param_from_file_to_host(param_path.c_str()));
        } else {
            RETURN_STATUS(model->executor->load_param_zeros_to_host());
        }
        auto idx = model_pool_size.fetch_add(1);
        if (idx >= model_pool_capacity) {
            LOG(ERROR) << "model pool is full";
            RETURN_STATUS(Status::Fail);
        }
        //model->executor.set_preempt_flag(preempt_flag);
        model_pool[idx] = std::move(model);
        LOG(INFO) << "load model from " << json_path << ", idx: " << idx;
        mid = idx;
        return Status::Succ;
    }

    Status BaseScheduler::load_model_from_host_to_device(ModelID& mid) {
        LOG(INFO) << "Base scheduler's load_model_from_host_to_device() has not been implemented!";
        return Status::Fail;
    }

    Status BaseScheduler::request_launch_kernel(int launch_kernel_queue_id, bool is_real_time, int op_offset,
                                                int kernel_offset, int TPC_request_num_cores,
                                                int TPC_limit_num_cores, double launch_progress,
                                                double runtime_ms) {
        std::shared_ptr<missilebase::utils::GPULaunchKernelRequest> request =
                                     std::make_shared<missilebase::utils::GPULaunchKernelRequest>(launch_kernel_queue_id,
                                                                                           op_offset, kernel_offset, 
                                                                                           TPC_request_num_cores, 
                                                                                           TPC_limit_num_cores, 
                                                                                           launch_progress, 
                                                                                           runtime_ms);
        if(is_real_time) {
            ASSERT_MSG(launch_kernel_queue_id < rt_task_gpu_launch_kernel_queues.size(), "Invalid launch_kernel_queue_id: " 
                                                  << launch_kernel_queue_id 
                                                  << " / " << rt_task_gpu_launch_kernel_queues.size());
            ASSERT_MSG(rt_task_gpu_launch_kernel_queues[launch_kernel_queue_id] != nullptr, "Invalid launch_kernel_queue_id: "
                                                  << launch_kernel_queue_id << " / " 
                                                  << rt_task_gpu_launch_kernel_queues.size());
            rt_task_gpu_launch_kernel_queues[launch_kernel_queue_id]->push(request);
        } else {
            ASSERT_MSG(launch_kernel_queue_id < be_task_gpu_launch_kernel_queues.size(), "Invalid launch_kernel_queue_id: " 
                                                   << launch_kernel_queue_id << " / " 
                                                   << be_task_gpu_launch_kernel_queues.size());
            ASSERT_MSG(be_task_gpu_launch_kernel_queues[launch_kernel_queue_id] != nullptr, "Invalid launch_kernel_queue_id: " 
                                                   << launch_kernel_queue_id << " / " 
                                                   << be_task_gpu_launch_kernel_queues.size());
            be_task_gpu_launch_kernel_queues[launch_kernel_queue_id]->push(request);
        }
        return Status::Succ;
    }

    Status BaseScheduler::create_queue(
        const TaskQueueType& qtp,
        const int perc_sm_request,
        const int perc_sm_limit,
        const int alloc_memory_size_kib,
        const int perc_pcie_bandwidth,
        QueueID& qid
    ) {
        int num_TPCs = missilebase::executor::GPUConfig::get_num_TPCs();
        int TPC_request_num_cores = int((float)(perc_sm_request * num_TPCs) / 100.0);
        int TPC_limit_num_cores = int((float)(perc_sm_limit * num_TPCs) / 100.0);
        //std::shared_ptr<TaskQueue> q;
        //RETURN_STATUS(create_task_queue(q, false, TPC_request_num_cores, TPC_limit_num_cores));
        if (qtp == TaskQueueType::RealTimeQueue) {
            // writer lock
            std::unique_lock<std::mutex> lock(rt_queues_mtx);
            auto idx = rt_queue_cnt;
            if (avail_rt_queue_ids.empty()) RETURN_STATUS(Status::Full);
            //rt_queues[idx] = std::move(q);
            rt_queue_cnt++;
            qid = avail_rt_queue_ids.top(); //idx;
            avail_rt_queue_ids.pop();
            RETURN_STATUS(create_task_queue(rt_queues[qid], 
                                            true, 
                                            TPC_request_num_cores, 
                                            TPC_limit_num_cores, 
                                            alloc_memory_size_kib, 
                                            perc_pcie_bandwidth));
        } else {
            // writer lock
            std::unique_lock<std::mutex> lock(be_queues_mtx);
            auto idx = be_queue_cnt;
            if (avail_be_queue_ids.empty()) RETURN_STATUS(Status::Full);
            //be_queues[idx] = std::move(q);
            be_queue_cnt++;
            qid = avail_be_queue_ids.top();
            avail_be_queue_ids.pop();
            RETURN_STATUS(create_task_queue(be_queues[qid], 
                          false, 
                          TPC_request_num_cores, 
                          TPC_limit_num_cores, 
                          alloc_memory_size_kib, 
                          perc_pcie_bandwidth));
        }
        LOG(INFO) << "Create task queue #" << qid << ", SM Request = " 
                  << perc_sm_request << "%, SM Limit = " 
                  << perc_sm_limit << "%" << std::endl;
        return Status::Succ;
    }

    Status BaseScheduler::bind_model_queue(
        const TaskQueueType& type,
        const QueueID& qid,
        const ModelID& mid
    ) {
        LOG(INFO) << "Base scheduler's bind_model_queue() has not been implemented!";
        return Status::Fail;
    }

    Status BaseScheduler::new_task(const ModelID& mid, TaskID& tid) {
        if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
        auto &model = model_pool[mid];
        std::shared_ptr<Task> task(new Task);
        task->model = model;
        task->id = task_idx_pool.fetch_add(1);
        task->qid = model->qid;
        task->create_step_offset = 0;
        task->swap_in_step_offset = 0;
        task->swap_out_step_offset = 0;
        task->launch_step_offset = 0;
        task->reclaim_step_offset = 0;
        task->state = TaskState::Init;
        task->submit = std::chrono::system_clock::now();
        task->preempted = false;
        task->padding = false;
        task->padding_to_finish = false;
        if (model->is_real_time) {
            num_executing_rt_tasks++;
            rt_queues[model->qid]->task_queue.push(task);
        } else {
            be_queues[model->qid]->task_queue.push(task);
        }
        tid = task->id;
        {
            std::unique_lock<std::mutex> lock(task_cnt_mtx);
            if (task_cnt == 0)
                task_cnt_cv.notify_all();
            task_cnt++;
        }
        {
            std::unique_lock<std::mutex> lock(task_pool_mtx);
            task_pool.insert({tid, task});
        }
        return Status::Succ;
    }

    Status BaseScheduler::get_data_size(ModelID mid, const std::string& name, size_t& size) {
        if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
        auto &model = model_pool[mid];
        return model->executor->get_data_size(name, size);
    }


    Status BaseScheduler::set_input(ModelID mid, const void* data, size_t len, const std::string& name) {
        if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
        auto &model = model_pool[mid];
        return model->executor->set_input(name, data, len);
    }

    Status BaseScheduler::get_output(ModelID mid, void* data, size_t len, const std::string& name) {
       if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
       auto &model = model_pool[mid];
       return model->executor->get_output(data, len);
    }


    Status BaseScheduler::run() {
        LOG(INFO) << "Base scheduler's run() has not been implemented!" << std::endl;
        return Status::Fail;
    }

    Status BaseScheduler::shutdown() {
        LOG(INFO) << "Base scheduler's shutdown() has not been implemented!" << std::endl;
        return Status::Succ;
    }

    Status BaseScheduler::logout_model(ModelID mid) {
        LOG(INFO) << "Base scheduler's logout_model() has not been implemented!" << std::endl;
        return Status::Succ;
    }

    ScheduleMode BaseScheduler::sche_mode() const {
        return mode;
    }

    bool BaseScheduler::rt_queues_task_empty(){
        for(int i=0;i<rt_queues.size();i++) {
            if (!rt_queues[i]->task_queue.empty()) {
                return false;
            }
        }
        return true;
    }

    int64_t BaseScheduler::avg_preempt_latency() const {
        if (preempt_count == 0) return 0;
        return preempt_latency_sum / preempt_count;
    }

    int64_t BaseScheduler::avg_kernel_sel_latency() const {
        if (kernel_sel_count == 0) return 0;
        return kernel_sel_latency_sum / kernel_sel_count;
    }

    void BaseScheduler::preempt_be_tasks(int num_required_TPCs) {
        LOG(INFO) << "BaseScheduler's preempt_be_tasks() has not been implemented!";
    }

    double BaseScheduler::get_memory_capacity_gib(bool is_real_time, int qid) {
        if(is_real_time) {
            ASSERT_MSG(qid>=0 && qid<this->rt_queues.size(), "Invalid RT queue id: " << qid);
            ASSERT_MSG(this->rt_queues[qid] != nullptr, "Invalid RT queue: NULL pointer.");
            return ((double)this->rt_queues[qid]->alloc_memory_size_kib) / (1024.0*1024.0);
        } else {
            ASSERT_MSG(qid>=0 && qid<this->be_queues.size(), "Invalid BE queue id: " << qid);
            ASSERT_MSG(this->be_queues[qid] != nullptr, "Invalid BE queue: NULL pointer.");
            return ((double)this->be_queues[qid]->alloc_memory_size_kib) / (1024.0*1024.0);
        }
    }

    Status BaseScheduler::get_task(TaskID tid, std::shared_ptr<Task>& t) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(task_pool_mtx);
            auto res = task_pool.find(tid);
            if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
            task = res->second;
        }
        t = task;
        return Status::Succ;
    }

    Status BaseScheduler::wait_task(TaskID tid, TaskState &response) {
        std::shared_ptr<Task> task;
        RETURN_STATUS(get_task(tid, task));
        #ifdef ENABLE_TASK_CV
                {
                    std::unique_lock<std::mutex> lock(task->mtx);
                    while (task->state != TaskState::Finish && task->state != TaskState::Reject) {
                        task->cv.wait(lock);
                    }
                }
        #else
                while (task->state != TaskState::Finish && task->state != TaskState::Reject) {
                    usleep(10);
                }
        #endif
        response = task->state;
        // {
        //     std::unique_lock<std::mutex> lock(task_pool_mtx);
        //     auto res = task_pool.find(tid);
        //     if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
        //     task_pool.erase(res);
        // }
        return Status::Succ;
    }
} // namespace server
} // namespace missilebase