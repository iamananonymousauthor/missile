#include <iomanip>

#include "../scheduler_utils.h"
#include "../base_scheduler.h"
#include "naive_scheduler.h"
#include "../../executor/naive/naive_executor.h"
#include "util/common.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#include <cuda.h>
#endif

#define ENABLE_TASK_CV 

namespace missilebase {
namespace server {

using namespace missilebase::executor;

NaiveScheduler::NaiveScheduler(int _gpu_id, bool _record_latency_per_kernel) :
    BaseScheduler(_gpu_id, Default, 1, 0) {
    LOG(INFO) << "NaiveScheduler: GPU ID = " << _gpu_id;
    task_thread = std::make_unique<std::thread>();
    record_latency_per_kernel = _record_latency_per_kernel;
}

NaiveScheduler::~NaiveScheduler() {

}

Status NaiveScheduler::load_model_from_host_to_device(ModelID& mid) {
    if(model_pool[mid].get() == nullptr) {
        LOG(ERROR) << "Invalid model [" << mid << "] (nullptr).";
        return Status::Fail;
    }
    return model_pool[mid]->executor->load_param_from_host_to_device();
}

Status NaiveScheduler::bind_model_queue(const TaskQueueType& type,
                                        const QueueID& qid,
                                        const ModelID& mid) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    //if(type == RealTimeQueue && qid >= max_num_rt_queues) RETURN_STATUS(Status::OutOfRange);
    //if(type == BestEffortQueue && qid >= max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
    //if (max_num_rt_queues + be_queue_cnt <= qid && qid < max_num_be_queues) RETURN_STATUS(Status::OutOfRange);
    if(record_latency_per_kernel) {
        std::dynamic_pointer_cast<NaiveExecutor>(model_pool[mid]->executor)->setup_kernel_latency_recorder();
    }
    model_pool[mid]->qid = qid;
    return Status::Succ;
}

Status NaiveScheduler::run() {
    LOG(INFO) << "Scheduler is ready for serving requests.";
    task_thread.reset(new std::thread([this] () {
        ASSERT_GPU_ERROR(GPUCtxSetCurrent(missilebase::executor::GPUConfig::get_gpu_ctx()));
        // make sure the real_time queue is created for convenience.
        while (true) {
            this->loop_body_thread(rt_queues[0]);
            if (this->_shutdown.load()) return;
        }
    }));
    return Status::Succ;
}

Status NaiveScheduler::shutdown() {
    _shutdown.store(true);
    task_thread->join();
    return Status::Succ;
}

void NaiveScheduler::loop_body_thread(std::shared_ptr<TaskQueue>& tqueue) {
    if (!tqueue->task_queue.empty()) {
        auto rt_task = tqueue->task_queue.front();
        rt_task->state = TaskState::Executing;
        execute_task(rt_task, tqueue);
        tqueue->task_queue.pop();
    }
}

void NaiveScheduler::execute_task(std::shared_ptr<Task> &task, std::shared_ptr<TaskQueue> &task_queue) {
    LOG(INFO) << "start rt task";
    task->state = TaskState::Executing;
    task->start = std::chrono::system_clock::now();
    task->model->executor->execute(task->create_step_offset, 
                                   task->swap_in_step_offset, 
                                   task->swap_out_step_offset,
                                   task->launch_step_offset, 
                                   task->reclaim_step_offset, 
                                   task_queue->stream,
                                   task_queue->TPC_request_num_cores, 
                                   task_queue->TPC_limit_num_cores);
    task->end = std::chrono::system_clock::now();
    task->state = TaskState::Finish;
    #ifdef ENABLE_TASK_CV
        {
            std::unique_lock<std::mutex> lock(task->mtx);
            task->cv.notify_all();
        }
    #endif
    auto now = std::chrono::system_clock::now();
    LOG(INFO) << "rt task finish, time elapsed = " 
              << std::chrono::duration_cast<std::chrono::microseconds>(task->end - task->start).count() 
              << ", " << std::chrono::duration_cast<std::chrono::microseconds>(now - task->start).count() 
              << "microseconds";
    return;
}

} // namespace server
} // namespace missilebase