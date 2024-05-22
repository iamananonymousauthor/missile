#include <iomanip>

#include "../scheduler_utils.h"
#include "../base_scheduler.h"
#include "temporal_multiplexing_scheduler.h"
#include "../../executor/temporal_multiplexing/temporal_multiplexing_executor.h"
#include "util/common.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#include <cuda.h>
#endif

#define ENABLE_TASK_CV 
//#define DEBUG_MODE

namespace missilebase {
namespace server {

using namespace missilebase::executor;

TemporalMultiplexingScheduler::TemporalMultiplexingScheduler(int _gpu_id, 
                                                             ScheduleMode _mode, 
                                                             int _max_num_rt_queues, 
                                                             int _max_num_be_queues) :
    rt_task_thread_pool(_max_num_rt_queues),
    be_task_thread_pool(_max_num_be_queues),
    BaseScheduler(_gpu_id, _mode, _max_num_rt_queues, _max_num_be_queues) {
    LOG(INFO) << "TemporalMultiplexingScheduler: GPU ID = " << _gpu_id;

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
}

TemporalMultiplexingScheduler::~TemporalMultiplexingScheduler() {

}

Status TemporalMultiplexingScheduler::load_model_from_host_to_device(ModelID& mid) {
    if(model_pool[mid].get() == nullptr) {
        LOG(ERROR) << "Invalid model [" << mid << "] (nullptr).";
        return Status::Fail;
    }
    LOG(INFO) << "DEBUG: LOAD FROM HOST TO DEVICE: model [" << mid << "].";
    return model_pool[mid]->executor->load_param_from_host_to_device();
}

Status TemporalMultiplexingScheduler::bind_model_queue( const TaskQueueType& type,
                                                        const QueueID& qid,
                                                        const ModelID& mid) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    if(type == RealTimeQueue && qid >= max_num_rt_queues) RETURN_STATUS(Status::OutOfRange);
    if(type == BestEffortQueue && qid >= max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
    //if (max_num_rt_queues + be_queue_cnt <= qid && qid < max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
    model_pool[mid]->qid = qid;
    return Status::Succ;
}

Status TemporalMultiplexingScheduler::run_naive_temporal() {
    for(int i=0; i<rt_task_thread_pool.size();i++) {
        rt_task_thread_pool[i] = std::make_unique<std::thread>();
        rt_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            // make sure the real_time queue is created for convenience.
            while (true) {
                this->loop_body_rt_thread(queue_idx, rt_queues[queue_idx]);
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
                this->loop_body_be_thread(queue_idx, be_queues[queue_idx]);
                if (this->_shutdown.load()) return;
            }
        }, i));
    }
}

Status TemporalMultiplexingScheduler::run_clockwork() {
    for(int i=0; i<rt_task_thread_pool.size();i++) {
        rt_task_thread_pool[i] = std::make_unique<std::thread>();
        rt_task_thread_pool[i].reset(new std::thread([this] (int queue_idx) {
            ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
            // make sure the real_time queue is created for convenience.
            while (true) {
                this->loop_body_rt_thread_clockwork(queue_idx, rt_queues[queue_idx]);
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
                this->loop_body_be_thread(queue_idx, be_queues[queue_idx]);
                if (this->_shutdown.load()) return;
            }
        }, i));
    }
}

Status TemporalMultiplexingScheduler::run() {
    //if (scheduler.get() != nullptr) RETURN_STATUS(Status::Fail);
    /*if (rt_queue.get() == nullptr) {
        assert(create_task_queue(rt_queue, true) == Status::Succ);
    }*/
    LOG(INFO) << "Scheduler is ready for serving requests. Mode = " << this->mode;
    if (this->mode == WaitPreempt) {
        return this->run_naive_temporal();
    } else if (this->mode == ClockworkPolicy) {
        return this->run_clockwork();
    }
}

Status TemporalMultiplexingScheduler::shutdown() {
    _shutdown.store(true);
    for(int i=0; i<rt_task_thread_pool.size();i++) {
        rt_task_thread_pool[i]->join();
    }
    for(int i=0; i<be_task_thread_pool.size();i++) {
        be_task_thread_pool[i]->join();
    }
    return Status::Succ;
}

void TemporalMultiplexingScheduler::loop_body_rt_thread(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
    if (!tqueue->task_queue.empty()) {
        auto rt_task = tqueue->task_queue.front();
        rt_task->state = TaskState::Executing;
        execute_rt_task(rt_task, tqueue);
        tqueue->task_queue.pop();
    }
}

void TemporalMultiplexingScheduler::loop_body_be_thread(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
    if (!tqueue->task_queue.empty()) {
        auto be_task = tqueue->task_queue.front();
        be_task->state = TaskState::Executing;
        execute_be_task(be_task, tqueue);
        if(be_task->is_finished()) {
            tqueue->task_queue.pop();
        }
    }
}

void TemporalMultiplexingScheduler::loop_body_rt_thread_clockwork(int queue_id, std::shared_ptr<TaskQueue>& tqueue) {
    if (!tqueue->task_queue.empty()) {
        auto rt_task = tqueue->task_queue.front();
        LOG(INFO) << "DEBUG: rt loop "<< queue_id << " handle a task: start";
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(now - rt_task->submit).count() + 
                                                        rt_task->model->estimated_runtime_ms >= rt_task->model->slo_ms) {
            rt_task->state = TaskState::Reject;
            LOG(INFO) << "DEBUG: rt reject "<< queue_id;
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
        LOG(INFO) << "DEBUG: rt loop "<< queue_id << " handle a task: finish";
    }
}

void TemporalMultiplexingScheduler::preempt_be_tasks(int num_required_TPCs, 
                                                     std::shared_ptr<Task> &be_task_to_be_preempted, 
                                                     GPUTPCAllocatedMap_t &TPCs_return_allocated) {
    #ifdef DEBUG_MODE
        LOG(INFO) << "DEBUG: preempt";
    #endif
    auto start = std::chrono::system_clock::now();
    if(GPUConfig::try_allocate_TPCs(num_required_TPCs, TPCs_return_allocated) ==
       Status::Succ) {
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
                << num_required_TPCs << " / be_queue[" << i << "] / be_task_used_num_TPCs = " << be_task_used_num_TPCs;
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

void TemporalMultiplexingScheduler::resume_be_tasks(std::shared_ptr<Task> &be_task_to_be_preempted) {
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

void TemporalMultiplexingScheduler::execute_rt_task(std::shared_ptr<Task> &task, std::shared_ptr<TaskQueue> &rt_queue) {
    #ifdef DEBUG_MODE
        LOG(INFO) << "DEBUG: start rt task";
    #endif
    task->state = TaskState::Executing;
    task->start = std::chrono::system_clock::now();
    // auto &exe = task->model->executor;
    // exe.execute(rt_queue->stream);
    GPUTPCAllocatedMap_t TPCs_return_allocated;
    std::shared_ptr<Task> be_task_to_be_preempted;
    be_task_to_be_preempted = nullptr;
    preempt_be_tasks(GPUConfig::get_num_TPCs(), be_task_to_be_preempted, TPCs_return_allocated);
    if (mode == ClockworkPolicy && std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - 
                                task->submit).count() + task->model->estimated_runtime_ms >= task->model->slo_ms) {
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

void TemporalMultiplexingScheduler::execute_be_task(std::shared_ptr<Task>& task, 
                                                    std::shared_ptr<TaskQueue>& be_queue) {
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