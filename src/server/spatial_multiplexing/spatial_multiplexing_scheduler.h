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
#include "../../executor/spatial_multiplexing/spatial_multiplexing_executor.h"
#include "../../cuda_implementation/cuda_impl.h"
#include "../../util/kernel_profile.h"
#include "../../util/launch_kernel_queue.h"

#define COMPUTE_BOUND_SM_UTILIZATION_THRESHOLD 60 // Used in Orion (EuroSys'24)
#define MEMORY_BOUND_DRAM_THROUGHPUT_THRESHOLD 60 // Used in Orion (EuroSys'24)

namespace missilebase {
namespace server {

using namespace missilebase::utils;

class SpatialMultiplexingScheduler : public BaseScheduler {
public:
    SpatialMultiplexingScheduler(int _gpu_id, ScheduleMode _mode, 
                                 int _max_num_rt_queues = 32, 
                                 int _max_num_be_queues = 4);
    ~SpatialMultiplexingScheduler();

    Status load_model_from_host_to_device(ModelID& mid);
    Status bind_model_queue(
        const TaskQueueType& type,
        const QueueID& qid,
        const ModelID& mid
    );
    Status run();
    Status shutdown();
    void preempt_be_tasks(int num_required_TPCs, std::shared_ptr<Task> &be_task_to_be_preempted, 
                          GPUTPCAllocatedMap_t &TPCs_return_allocated);
    void resume_be_tasks(std::shared_ptr<Task> &be_task_to_be_preempted);

    Status wait_launch_kernel_complete(int launch_kernel_queue_id, bool is_real_time);

private:
    std::mutex preempt_mtx; // Only one rt task is allowed to select which be task to be preempted at the same moment.

    std::vector<std::unique_ptr<std::thread> > rt_task_thread_pool, be_task_thread_pool;

    std::unique_ptr<std::thread> gpu_launch_rt_kernel_thread, gpu_launch_be_kernel_thread;

    executor::GPUStream_t rt_gpu_launch_kernel_stream, be_gpu_launch_kernel_stream;

    int last_selected_rt_qid, last_selected_be_qid;

    int orion_sm_threshold;

    std::atomic<int> has_rt_task_running;
    //std::atomic<bool> has_rt_kernel_running;
    KernelProfile rt_kernel_profile;
    std::atomic<double> running_rt_task_total_latency;
    double DUR_THRESHOLD;

    void execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);
    void execute_rt_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);

    Status run_multitask();
    Status run_orion();

    bool schedule_be_orion(KernelProfile rt_kernel_profile, KernelProfile be_kernel_profile);
    void loop_body_gpu_launch_rt_kernel_orion();
    void loop_body_gpu_launch_be_kernel_orion();

    void loop_body_rt_thread_multitask(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
    void loop_body_rt_thread_orion(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
    void loop_body_be_thread_multitask(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
    void loop_body_be_thread_orion(int queue_id, std::shared_ptr<TaskQueue>& tqueue);
};


} // namespace server
} // namespace missilebase