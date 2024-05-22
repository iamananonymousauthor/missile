#pragma once

#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>

#include "util/threadsafe_queue.h"
#include "util/common.h"
#include "../base_scheduler.h"
#include "../scheduler_utils.h"
#include "../../executor/temporal_multiplexing/temporal_multiplexing_executor.h"
#include "../../cuda_implementation/cuda_impl.h"

namespace missilebase {
namespace server {

class TemporalMultiplexingScheduler : public BaseScheduler {
public:
    TemporalMultiplexingScheduler(int _gpu_id, ScheduleMode _mode, 
                                  int _max_num_rt_queues = 32, 
                                  int _max_num_be_queues = 32);
    ~TemporalMultiplexingScheduler();

    Status load_model_from_host_to_device(ModelID& mid);
    Status bind_model_queue(
        const TaskQueueType& type,
        const QueueID& qid,
        const ModelID& mid
    );
    Status run();
    Status shutdown();
    void preempt_be_tasks(int num_required_TPCs, 
                          std::shared_ptr<Task> &be_task_to_be_preempted, 
                          GPUTPCAllocatedMap_t &TPCs_return_allocated);
    void resume_be_tasks(std::shared_ptr<Task> &be_task_to_be_preempted);

private:
    std::mutex preempt_mtx; // Only one rt task is allowed to select which be task to be preempted at the same moment.

    std::vector<std::unique_ptr<std::thread> > rt_task_thread_pool, be_task_thread_pool;

    void execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);
    void execute_rt_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);

    Status run_naive_temporal();
    Status run_clockwork();

    void loop_body_rt_thread(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
    void loop_body_rt_thread_clockwork(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
    void loop_body_be_thread(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
};


} // namespace server
} // namespace missilebase