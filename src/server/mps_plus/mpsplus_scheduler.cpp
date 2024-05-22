#include <iomanip>

#include "../scheduler_utils.h"
#include "../base_scheduler.h"
#include "mpsplus_scheduler.h"
#include "../../executor/mpsplus_multiplexing/mpsplus_executor.h"
#include "util/common.h"
#include "../../util/kernel_profile.h"
#include "../../util/launch_kernel_queue.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#include <cuda.h>
#endif

#define ENABLE_TASK_CV 
//#define DEBUG_MODE

namespace missilebase {
namespace server {

    using namespace missilebase::executor;
    using namespace missilebase::utils;

    MPSPlusScheduler::MPSPlusScheduler(int _gpu_id, ScheduleMode _mode, int _max_num_rt_queues, int _max_num_be_queues) :
        rt_task_thread_pool(_max_num_rt_queues),
        be_task_thread_pool(_max_num_be_queues),
        BaseScheduler(_gpu_id, _mode, _max_num_rt_queues, _max_num_be_queues) {
        LOG(INFO) << "MPSPlusScheduler: GPU ID = " << _gpu_id;
        memcpy_htod_queues[0].resize(be_task_thread_pool.size());
        memcpy_htod_queues[1].resize(rt_task_thread_pool.size());
        memcpy_dtoh_queues[0].resize(be_task_thread_pool.size());
        memcpy_dtoh_queues[1].resize(rt_task_thread_pool.size());
        for(int i=0; i<rt_task_thread_pool.size();i++) {
            memcpy_htod_queues[1][i] = std::make_shared<GPULaunchMemcpyQueue>();
            memcpy_dtoh_queues[1][i] = std::make_shared<GPULaunchMemcpyQueue>();
        }
        for(int i=0; i<be_task_thread_pool.size();i++) {
            memcpy_htod_queues[0][i] = std::make_shared<GPULaunchMemcpyQueue>();
            memcpy_dtoh_queues[0][i] = std::make_shared<GPULaunchMemcpyQueue>();
        }

        for(int i=0; i<rt_task_thread_pool.size();i++) {
            rt_task_thread_pool[i] = std::make_unique<std::thread>();
        }
        for(int i=0; i<be_task_thread_pool.size();i++) {
            be_task_thread_pool[i] = std::make_unique<std::thread>();
        }

        for(int i=0; i<max_num_rt_queues;i++) {
            ASSERT_GPU_ERROR(GPUStreamCreate(&rt_queues[i]->stream));
        }
        for(int i=0; i<max_num_be_queues;i++) {
            ASSERT_GPU_ERROR(GPUStreamCreate(&be_queues[i]->stream));
        }

        last_selected_be_qid = 0;
        orion_sm_threshold = GPUConfig::get_num_TPCs()*2; // NOTE: This is the default value, according to Orion's paper.
        std::atomic_init(&has_rt_task_running, 0);
        std::atomic_init(&running_rt_task_total_latency, 0);
        DUR_THRESHOLD = 0.025; // NOTE: This is the default value (2.5%), according to Orion's paper.
    }

    MPSPlusScheduler::~MPSPlusScheduler() {

    }

    Status MPSPlusScheduler::load_model_from_host_to_device(ModelID& mid) {
        if(model_pool[mid].get() == nullptr) {
            LOG(ERROR) << "Invalid model [" << mid << "] (nullptr).";
            return Status::Fail;
        }
        return model_pool[mid]->executor->load_param_from_host_to_device();
    }

    Status MPSPlusScheduler::bind_model_queue(
            const TaskQueueType& type,
            const QueueID& qid,
            const ModelID& mid
    ) {
        if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
        if(type == RealTimeQueue && qid >= max_num_rt_queues) RETURN_STATUS(Status::OutOfRange);
        if(type == BestEffortQueue && qid >= max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
        //if (max_num_rt_queues + be_queue_cnt <= qid && qid < max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
        model_pool[mid]->qid = qid;
        return Status::Succ;
    }

    Status MPSPlusScheduler::run_multitask() {
        for(int i=0; i<rt_task_thread_pool.size();i++) {
            rt_task_thread_pool[i] = std::make_unique<std::thread>();
            rt_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                // make sure the real_time queue is created for convenience.
                while (true) {
                    this->loop_body_rt_thread_multitask(queue_idx, rt_queues[queue_idx]);
                    if (this->_shutdown.load()) return;
                }
            }, i));
        }

        for(int i=0; i<be_task_thread_pool.size();i++) {
            be_task_thread_pool[i] = std::make_unique<std::thread>();
            be_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                // make sure the real_time queue is created for convenience.
                while (true) {
                    this->loop_body_be_thread_multitask(queue_idx, be_queues[queue_idx]);
                    if (this->_shutdown.load()) return;
                }
            }, i));
        }
    }

    Status MPSPlusScheduler::run_orion() {
        gpu_launch_rt_kernel_thread = std::make_unique<std::thread>();
        gpu_launch_rt_kernel_thread.reset(new std::thread([this] () {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            ASSERT_GPU_ERROR(GPUStreamCreateWithPriority(&rt_gpu_launch_kernel_stream, 0));
            while (true) {
                this->loop_body_gpu_launch_rt_kernel_orion();
                if (this->_shutdown.load()) return;
            }
        }));

        gpu_launch_be_kernel_thread = std::make_unique<std::thread>();
        gpu_launch_be_kernel_thread.reset(new std::thread([this] () {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            ASSERT_GPU_ERROR(GPUStreamCreateWithPriority(&be_gpu_launch_kernel_stream, 1));
            while (true) {
                this->loop_body_gpu_launch_be_kernel_orion();
                if (this->_shutdown.load()) return;
            }
        }));

        gpu_launch_memcpy_htod_thread = std::make_unique<std::thread>();
        gpu_launch_memcpy_htod_thread.reset(new std::thread([this] () {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            ASSERT_GPU_ERROR(GPUStreamCreateWithPriority(&memcpy_htod_stream, 0));
            while (true) {
                this->loop_body_htod_copy();
                if (this->_shutdown.load()) return;
            }
        }));

        gpu_launch_memcpy_dtoh_thread = std::make_unique<std::thread>();
        gpu_launch_memcpy_dtoh_thread.reset(new std::thread([this] () {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            ASSERT_GPU_ERROR(GPUStreamCreateWithPriority(&memcpy_dtoh_stream, 0));
            while (true) {
                this->loop_body_dtoh_copy();
                if (this->_shutdown.load()) return;
            }
        }));

        for(int i=0; i<rt_task_thread_pool.size();i++) {
            rt_task_thread_pool[i] = std::make_unique<std::thread>();
            rt_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                // make sure the real_time queue is created for convenience.
                while (true) {
                    this->loop_body_rt_thread_orion(queue_idx, rt_queues[queue_idx]);
                    if (this->_shutdown.load()) return;
                }
            }, i));
        }

        for(int i=0; i<be_task_thread_pool.size();i++) {
            be_task_thread_pool[i] = std::make_unique<std::thread>();
            be_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
                ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
                // make sure the real_time queue is created for convenience.
                while (true) {
                    this->loop_body_be_thread_orion(queue_idx, be_queues[queue_idx]);
                    if (this->_shutdown.load()) return;
                }
            }, i));
        }
    }

    Status MPSPlusScheduler::run() {
        //if (scheduler.get() != nullptr) RETURN_STATUS(Status::Fail);
        /*if (rt_queue.get() == nullptr) {
            assert(create_task_queue(rt_queue, true) == Status::Succ);
        }*/
        LOG(INFO) << "Scheduler is ready for serving requests. Mode = " << this->mode;
        if (this->mode == MPSPlusPolicy) {
            return this->run_orion();
        }
    }

    Status MPSPlusScheduler::shutdown() {
        _shutdown.store(true);
        for(int i=0; i<rt_task_thread_pool.size();i++) {
            rt_task_thread_pool[i]->join();
        }
        for(int i=0; i<be_task_thread_pool.size();i++) {
            be_task_thread_pool[i]->join();
        }
        return Status::Succ;
    }

    void MPSPlusScheduler::loop_body_rt_thread_multitask(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
        if (!tqueue->task_queue.empty()) {
            auto rt_task = tqueue->task_queue.front();
            rt_task->state = TaskState::Executing;
            execute_rt_task(rt_task, tqueue);
            tqueue->task_queue.pop();
        }
    }

    void MPSPlusScheduler::loop_body_be_thread_multitask(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
        if (!tqueue->task_queue.empty()) {
            auto be_task = tqueue->task_queue.front();
            be_task->state = TaskState::Executing;
            execute_be_task(be_task, tqueue);
            if(be_task->is_finished()) {
                tqueue->task_queue.pop();
            }
        }
    }

    bool MPSPlusScheduler::schedule_be_orion(
            KernelProfile rt_kernel_profile, KernelProfile be_kernel_profile
            ) {
        if(be_kernel_profile.sm_needed >= orion_sm_threshold) {
            return false;
        }

        bool rt_is_memory_bound = rt_kernel_profile.dram_throughput_perc > MEMORY_BOUND_DRAM_THROUGHPUT_THRESHOLD;
        bool rt_is_compute_bound = rt_kernel_profile.compute_sm_perc > COMPUTE_BOUND_SM_UTILIZATION_THRESHOLD;

        bool be_is_memory_bound = be_kernel_profile.dram_throughput_perc > MEMORY_BOUND_DRAM_THROUGHPUT_THRESHOLD;
        bool be_is_compute_bound = be_kernel_profile.compute_sm_perc > COMPUTE_BOUND_SM_UTILIZATION_THRESHOLD;

        if(rt_kernel_profile.duration_ms < be_kernel_profile.duration_ms) {
            return false;
        } else if(rt_is_compute_bound && rt_is_memory_bound) {
            return false;
        } else if(!rt_is_memory_bound && !rt_is_compute_bound) {
            return true;
        } else if(!rt_is_memory_bound && !be_is_compute_bound) {
            return true;
        } else if(!rt_is_compute_bound && !be_is_memory_bound) {
            return true;
        } else {
            return false;
        }
    }

    void MPSPlusScheduler::loop_body_gpu_launch_rt_kernel_orion() {
        bool need_launch_rt_task = false;
        std::shared_ptr<GPULaunchKernelRequest> selected_rt_request;
        std::shared_ptr<Task> selected_rt_task;
        int selected_rt_qid;
        KernelProfile rt_kernel_profile_tmp;

        for(int round_idx=0; round_idx<rt_task_gpu_launch_kernel_queues.size(); round_idx++) {
            int qid = (round_idx + last_selected_rt_qid) % rt_task_gpu_launch_kernel_queues.size();
            if(rt_task_gpu_launch_kernel_queues[qid]->empty() == false) {
                selected_rt_request = rt_task_gpu_launch_kernel_queues[qid]->front();
                selected_rt_task = rt_queues[qid]->task_queue.front();
                std::shared_ptr<SpatialMultiplexingExecutor> ptr_to_executor = 
                    std::dynamic_pointer_cast<missilebase::server::SpatialMultiplexingExecutor>(selected_rt_task->model->executor);
                ASSERT_STATUS(ptr_to_executor->get_kernel_profile(selected_rt_request->op_offset, 
                                                                  selected_rt_request->kernel_offset, 
                                                                  rt_kernel_profile_tmp));
                need_launch_rt_task = true;
                selected_rt_qid = qid;
                break;
            }
        }

        if(need_launch_rt_task) {
            //has_rt_kernel_running.store(true);
            selected_rt_task->model->executor->launch_kernel(selected_rt_request->op_offset,
                                                            selected_rt_request->kernel_offset,
                                                            rt_gpu_launch_kernel_stream,
                                                            selected_rt_request->TPC_request_num_cores,
                                                            selected_rt_request->TPC_limit_num_cores);
            rt_task_gpu_launch_kernel_queues[selected_rt_qid]->pop();
            rt_kernel_profile = rt_kernel_profile_tmp;
            //ASSERT_GPU_ERROR(GPUStreamSynchronize(rt_gpu_launch_kernel_stream));
            //has_rt_kernel_running.store(false);
            rt_kernel_profile = KernelProfile();
            if(rt_task_gpu_launch_kernel_queues[selected_rt_qid]->empty()) {
                rt_task_gpu_launch_kernel_queues[selected_rt_qid]->notify_empty();
            }
            last_selected_rt_qid = selected_rt_qid;
        }
    }

    void MPSPlusScheduler::loop_body_gpu_launch_be_kernel_orion() {
        std::shared_ptr<GPULaunchKernelRequest> selected_be_request;
        std::shared_ptr<Task> selected_be_task;
        //KernelProfile be_kernel_profile;
        int selected_be_qid = -1;
        bool need_launch_be_task = false;
        for(int round_idx=0; round_idx<be_task_gpu_launch_kernel_queues.size(); round_idx++) {
            int qid = (round_idx + last_selected_be_qid) % be_task_gpu_launch_kernel_queues.size();
            if(be_task_gpu_launch_kernel_queues[qid]->empty() == false) {
                selected_be_request = be_task_gpu_launch_kernel_queues[qid]->front();
                selected_be_task = be_queues[qid]->task_queue.front();
                //std::shared_ptr<MPSPlusExecutor> ptr_to_executor = std::dynamic_pointer_cast<missilebase::server::MPSPlusExecutor>(selected_be_task->model->executor);
                //ASSERT_STATUS(ptr_to_executor->get_kernel_profile(selected_be_request->op_offset, selected_be_request->kernel_offset, be_kernel_profile));
                need_launch_be_task = true;
                selected_be_qid = qid;
                break;
            }
        }
        if(need_launch_be_task) {
            selected_be_task->model->executor->launch_kernel(selected_be_request->op_offset, 
                                                             selected_be_request->kernel_offset, 
                                                             be_gpu_launch_kernel_stream, 
                                                             selected_be_request->TPC_request_num_cores, 
                                                             selected_be_request->TPC_limit_num_cores);
            be_task_gpu_launch_kernel_queues[selected_be_qid]->pop();
            //ASSERT_GPU_ERROR(GPUStreamSynchronize(be_gpu_launch_kernel_stream));
            if(be_task_gpu_launch_kernel_queues[selected_be_qid]->empty()) {
                be_task_gpu_launch_kernel_queues[selected_be_qid]->notify_empty();
            }
            last_selected_be_qid = selected_be_qid;
        }
    }

    void MPSPlusScheduler::loop_body_htod_copy() {
        for(int i=0; i<rt_task_thread_pool.size();i++) {
            if (!memcpy_htod_queues[1][i]->empty()) {
                auto task = memcpy_htod_queues[1][i]->front();
                ASSERT_GPU_ERROR(GPUMemcpyHtoD(
                        (GPUDevicePtr_t) task->dst_ptr, (void *) task->src_ptr,
                        task->copy_size_bytes));
                memcpy_htod_queues[1][i]->pop();
                if(memcpy_htod_queues[1][i]->empty()) {
                    memcpy_htod_queues[1][i]->notify_empty();
                }
            }
        }
        for(int i=0; i<be_task_thread_pool.size();i++) {
            if (!memcpy_htod_queues[0][i]->empty()) {
                auto task = memcpy_htod_queues[0][i]->front();
                ASSERT_GPU_ERROR(GPUMemcpyHtoD((GPUDevicePtr_t) task->dst_ptr,
                                                 (void *) task->src_ptr,
                                                    task->copy_size_bytes));
                memcpy_htod_queues[0][i]->pop();
                if(memcpy_htod_queues[0][i]->empty()) {
                    memcpy_htod_queues[0][i]->notify_empty();
                }
            }
        }
    }

    void MPSPlusScheduler::loop_body_dtoh_copy() {
        for(int i=0; i<rt_task_thread_pool.size();i++) {
            if (!memcpy_dtoh_queues[1][i]->empty()) {
                auto task = memcpy_dtoh_queues[1][i]->front();
                ASSERT_GPU_ERROR(GPUMemcpyDtoH(
                        (void *) task->dst_ptr, (GPUDevicePtr_t) task->src_ptr,
                        task->copy_size_bytes));
                memcpy_dtoh_queues[1][i]->pop();
                if(memcpy_dtoh_queues[1][i]->empty()) {
                    memcpy_dtoh_queues[1][i]->notify_empty();
                }
            }
        }
        for(int i=0; i<be_task_thread_pool.size();i++) {
            if (!memcpy_dtoh_queues[0][i]->empty()) {
                auto task = memcpy_dtoh_queues[0][i]->front();
                ASSERT_GPU_ERROR(GPUMemcpyDtoH(
                        (void *) task->dst_ptr, (GPUDevicePtr_t) task->src_ptr,
                        task->copy_size_bytes));
                memcpy_dtoh_queues[0][i]->pop();
                if(memcpy_dtoh_queues[0][i]->empty()) {
                    memcpy_dtoh_queues[0][i]->notify_empty();
                }
            }
        }
    }

    void MPSPlusScheduler::loop_body_rt_thread_orion(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
        if (!tqueue->task_queue.empty()) {
            auto rt_task = tqueue->task_queue.front();
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            if(std::chrono::duration_cast<std::chrono::microseconds>(now - rt_task->submit).count() + 
                                                         rt_task->model->estimated_runtime_ms >= rt_task->model->slo_ms) {
                rt_task->state = TaskState::Reject;
                {
                    std::unique_lock <std::mutex> lock(rt_task->mtx);
                    rt_task->cv.notify_all();
                }
                tqueue->task_queue.pop();
            } else {
                rt_task->state = TaskState::Executing;
                execute_rt_task(rt_task, tqueue);
                tqueue->task_queue.pop();
            }
        }
    }

    void MPSPlusScheduler::loop_body_be_thread_orion(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
        if (!tqueue->task_queue.empty()) {
            auto be_task = tqueue->task_queue.front();
            be_task->state = TaskState::Executing;
            execute_be_task(be_task, tqueue);
            if(be_task->is_finished()) {
                tqueue->task_queue.pop();
            }
        }
    }

    void MPSPlusScheduler::preempt_be_tasks(int num_required_TPCs, 
                                            std::shared_ptr<Task> &be_task_to_be_preempted, 
                                            GPUTPCAllocatedMap_t &TPCs_return_allocated) {
        #ifdef DEBUG_MODE
            LOG(INFO) << "DEBUG: preempt";
        #endif
        auto start = std::chrono::system_clock::now();
        if(GPUConfig::try_allocate_TPCs(num_required_TPCs, TPCs_return_allocated) == Status::Succ) {
            return;
        }

        // Only one rt task is allowed to select which be task to be preempted at the same moment
        std::unique_lock<std::mutex> lock(preempt_mtx); 
        int used_num_TPCs_be_task_to_be_preempted = 10240;
        int available_TPCs = GPUConfig::get_number_of_available_TPCs();
        be_task_to_be_preempted = nullptr;

        for (int i = 0; i < be_task_thread_pool.size(); i++) {
            auto &be_queue = be_queues[i]->task_queue;
            if (!be_queue.empty()) {
                auto be_task = be_queue.front();
                int be_task_used_num_TPCs = be_task->model->executor->get_used_num_TPCs();
                #ifdef DEBUG_MODE
                            LOG(INFO) << "DEBUG: preempt_be_tasks: num_required_TPCs = " 
                            << num_required_TPCs << " / be_queue[" << i 
                            << "] / be_task_used_num_TPCs = " << be_task_used_num_TPCs;
                #endif
                if (be_task->state == TaskState::Executing &&
                    available_TPCs + be_task_used_num_TPCs >= num_required_TPCs &&
                    be_task_used_num_TPCs < used_num_TPCs_be_task_to_be_preempted) {
                    be_task_to_be_preempted = be_task;
                    used_num_TPCs_be_task_to_be_preempted = be_task_used_num_TPCs;
                }
            }
        }
        lock.unlock();

        if(be_task_to_be_preempted == nullptr) {
            #ifdef DEBUG_MODE
                    LOG(INFO) << "DEBUG: No valid BE task can be preempted";
            #endif
            return;
        }
        be_task_to_be_preempted->state = TaskState::Waiting;
        be_task_to_be_preempted->preempted = true;
        be_task_to_be_preempted->model->executor->TryToEvict(false);
        while (GPUConfig::try_allocate_TPCs(num_required_TPCs, TPCs_return_allocated) !=
            Status::Succ) {
            #ifdef DEBUG_MODE
                    LOG(INFO) << "DEBUG: wait for allocate TPCs: " << num_required_TPCs;
            #endif
            usleep(100);
        }
        auto end = std::chrono::system_clock::now();
        preempt_count++;
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        preempt_latency_sum += latency;
        #ifdef DEBUG_MODE
            LOG(INFO) << "preempt latency: " << latency << " us";
        #endif
    }

    void MPSPlusScheduler::resume_be_tasks(std::shared_ptr<Task> &be_task_to_be_preempted) {
        //if (preempted) return;
        #ifdef DEBUG_MODE
            LOG(INFO) << "DEBUG: resume be tasks";
        #endif
        auto start = std::chrono::system_clock::now();
        //be_task_to_be_preempted->preempted = false;
        be_task_to_be_preempted->model->executor->ClearEvictionFlag(false);
        auto end = std::chrono::system_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        #ifdef DEBUG_MODE
            LOG(INFO) << "resume latency: " << latency << " us";
        #endif
    }

    Status MPSPlusScheduler::wait_launch_kernel_complete(int launch_kernel_queue_id, bool is_real_time) {
        if(is_real_time) {
            rt_task_gpu_launch_kernel_queues[launch_kernel_queue_id]->wait_until_empty();
        } else {
            be_task_gpu_launch_kernel_queues[launch_kernel_queue_id]->wait_until_empty();
        }
        return Status::Succ;
    }

    void MPSPlusScheduler::execute_rt_task(std::shared_ptr<Task> &task, std::shared_ptr<TaskQueue> &rt_queue) {
        #ifdef DEBUG_MODE
            LOG(INFO) << "DEBUG: start rt task";
        #endif
        task->state = TaskState::Executing;
        task->start = std::chrono::system_clock::now();
        has_rt_task_running++;
        running_rt_task_total_latency.store(task->model->estimated_runtime_ms);
        // auto &exe = task->model->executor;
        // exe.execute(rt_queue->stream);
        GPUTPCAllocatedMap_t TPCs_return_allocated;
        std::shared_ptr<Task> be_task_to_be_preempted;
        be_task_to_be_preempted = nullptr;
        //preempt_be_tasks(GPUConfig::get_num_TPCs(), be_task_to_be_preempted, TPCs_return_allocated);
        if(mode == MultiTaskPolicy) {
            ASSERT_MSG(GPUConfig::try_allocate_TPCs(rt_queue->TPC_request_num_cores, TPCs_return_allocated) == Status::Succ,
                    "Invalid TPC request num cores: " << rt_queue->TPC_request_num_cores);
            task->model->executor->set_TPCs_allocated(TPCs_return_allocated);
        }
        if (mode == ClockworkPolicy && 
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - task->submit).count() + 
            task->model->estimated_runtime_ms >= task->model->slo_ms) {
            task->state = TaskState::Reject;
            #ifdef ENABLE_TASK_CV
                    {
                        std::unique_lock <std::mutex> lock(task->mtx);
                        task->cv.notify_all();
                    }
            #endif
            return;
        }
        task->model->executor->set_TPCs_allocated(TPCs_return_allocated);
        #ifdef DEBUG_MODE
            LOG(INFO) << "DEBUG: set_TPCs_allocated rt_queue->TPC_request_num_cores = " 
                      << rt_queue->TPC_request_num_cores << ", " << TPCs_return_allocated;
        #endif DEBUG_MODE
        task->model->executor->execute(task->create_step_offset, 
                                       task->swap_in_step_offset, 
                                       task->swap_out_step_offset,
                                       task->launch_step_offset, 
                                       task->reclaim_step_offset, 
                                       rt_queue->stream,
                                       GPUConfig::get_num_TPCs(), 
                                       GPUConfig::get_num_TPCs());
        task->end = std::chrono::system_clock::now();
        task->state = TaskState::Finish;
        GPUConfig::release_TPCs_in_map(TPCs_return_allocated);
        if(be_task_to_be_preempted != nullptr) {
            resume_be_tasks(be_task_to_be_preempted);
        }
        has_rt_task_running--;
        #ifdef ENABLE_TASK_CV
            {
                std::unique_lock<std::mutex> lock(task->mtx);
                task->cv.notify_all();
            }
        #endif
        #ifdef DEBUG_MODE
            auto now = std::chrono::system_clock::now();
            LOG(INFO) << "rt task finish, time elapsed = " 
            << std::chrono::duration_cast<std::chrono::microseconds>(now - task->start).count() 
            << "microseconds";
        #endif
        return;
    }

    void MPSPlusScheduler::execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& be_queue) {
        #ifdef DEBUG_MODE
            LOG(INFO) << "DEBUG: start be task";
        #endif
        task->state = TaskState::Executing;
        task->start = std::chrono::system_clock::now();
        // auto &exe = task->model->executor;
        // exe.execute(rt_queue->stream);
        task->model->executor->execute(task->create_step_offset, 
                                       task->swap_in_step_offset, 
                                       task->swap_out_step_offset,
                                       task->launch_step_offset, 
                                       task->reclaim_step_offset, 
                                       be_queue->stream,
                                       GPUConfig::get_num_TPCs(), 
                                       GPUConfig::get_num_TPCs());
        task->end = std::chrono::system_clock::now();
        if(task->is_finished()) {
            task->state = TaskState::Finish;
            #ifdef ENABLE_TASK_CV
                    {
                        std::unique_lock <std::mutex> lock(task->mtx);
                        task->cv.notify_all();
                    }
            #endif

            #ifdef DEBUG_MODE
                    LOG(INFO) << "be task finish";
            #endif
        }
        return;
    }
} // namespace server
} // namespace missilebase