#pragma once

#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <stack>
#include <thread>
#include <gdrapi.h>

#include "util/common.h"
#include "scheduler_utils.h"
#include "../util/launch_kernel_queue.h"

namespace missilebase {
namespace server {

class BaseScheduler {
public:
    ScheduleMode mode;
    std::shared_ptr<missilebase::utils::BuddyAllocator> rt_buddy_allocator, be_buddy_allocator, shared_buddy_allocator;
    std::atomic<int> num_executing_rt_tasks;
    // Only used in MPSPlus
    std::vector<std::shared_ptr<missilebase::utils::GPULaunchMemcpyQueue> > memcpy_htod_queues[2], memcpy_dtoh_queues[2];
    std::vector<std::shared_ptr<missilebase::utils::GPULaunchKernelQueue> >
                                              rt_task_gpu_launch_kernel_queues, be_task_gpu_launch_kernel_queues;
    CPUCoreManager cpu_core_manager;
    void* host_memory_pool = NULL;
    uint64_t host_memory_pool_capacity_bytes = 1024llu*1024llu*1024llu*20llu;
    uint64_t allocated_offset_bytes = 0;
    gdr_t gdr_g;
    gdr_mh_t gdr_mh;
    void *gdr_map_d_ptr  = NULL;
    void *gdr_buf_ptr = NULL;
    bool is_gdr_supported = false;

    BaseScheduler(int _gpu_id, 
                  ScheduleMode _mode = ScheduleMode::WaitPreempt, 
                  int _max_num_rt_queues = 64, 
                  int _max_num_be_queues = 16);
    virtual ~BaseScheduler();

    Status load_model_to_host(
        const SchedulerType scheduler_type,
        const std::string& kernel_co_path,
        const std::string& json_path,
        const std::string& param_path,
        ModelID& mid,
        bool _is_real_time,
        int _qid,
        int _estimated_runtime_ms,
        int _slo_ms
    );

    virtual Status load_model_from_host_to_device(ModelID& mid);

    Status create_queue(
        const TaskQueueType& qtp,
        const int perc_sm_request,
        const int perc_sm_limit,
        const int alloc_memory_size_kib,
        const int perc_pcie_bandwidth,
        QueueID& qid
    );

    virtual Status bind_model_queue(
        const TaskQueueType& type,
        const QueueID& qid,
        const ModelID& mid
    );

    Status get_data_size(ModelID mid, const std::string& name, size_t& size);

    Status set_input(ModelID mid, const void* data, size_t len, const std::string& name="data");

    Status get_output(ModelID mid, void* data, size_t len, const std::string& name="output");

    virtual Status run();
    virtual Status shutdown();

    virtual Status logout_model(ModelID mid);

    Status new_task(
        const ModelID& mid,
        TaskID& tid
    );

    Status wait_task(
        TaskID tid,
        TaskState &response
    );

    Status get_task(
        TaskID tid,
        std::shared_ptr<Task>& t
    );

    ScheduleMode sche_mode() const;

    void set_wait_sync(bool value);

    int64_t avg_preempt_latency() const;
    
    int64_t avg_kernel_sel_latency() const;

    virtual void preempt_be_tasks(int num_required_TPCs);

    double get_memory_capacity_gib(bool is_real_time, int qid);

    Status request_launch_kernel(int launch_kernel_queue_id, bool is_real_time, int op_offset,
                                 int kernel_offset, int TPC_request_num_cores,
                                 int TPC_limit_num_cores, double launch_progress, double runtime_ms);

protected:
    int gpu_id;
    DeviceType device_type;
    const size_t model_pool_capacity = 1024;
    std::atomic_uint32_t model_pool_size;
    std::vector<std::shared_ptr<TaskAbstractModel>> model_pool;


    std::atomic_uint32_t task_idx_pool;
    std::unordered_map<TaskID, std::shared_ptr<Task>> task_pool;
    std::mutex task_pool_mtx;

    size_t max_num_be_queues = 32, max_num_rt_queues = 32;
    //const QueueID rt_queue_id = 32; // the same with be queue num
    std::mutex be_queues_mtx;
    std::vector<std::shared_ptr<TaskQueue>> be_queues;
    std::stack<int> avail_be_queue_ids;
    volatile uint32_t be_queue_cnt;
    std::mutex rt_queues_mtx;
    std::vector<std::shared_ptr<TaskQueue>> rt_queues;
    std::stack<int> avail_rt_queue_ids;
    volatile uint32_t rt_queue_cnt;
    std::mutex task_cnt_mtx;
    std::condition_variable task_cnt_cv; // To wake up the scheduler
    volatile uint32_t task_cnt;
    bool wait_sync;

    executor::GPUStream_t execute_stream, preempt_stream;
    executor::GPUDevicePtr_t preempt_flag;
    executor::GPUContext_t gpu_ctx;

    std::mutex preempted_flag_mtx;
    bool preempted;

    int be_stream_device_queue_cap;
    std::atomic_bool _shutdown;

    uint64_t preempt_count;
    uint64_t preempt_latency_sum;

    uint64_t kernel_sel_count;
    uint64_t kernel_sel_latency_sum;

    Status create_task_queue(std::shared_ptr<TaskQueue>& ret, 
                             bool rt, 
                             int TPC_request_num_cores, 
                             int TPC_limit_num_cores, 
                             int alloc_memory_size_kib, 
                             int perc_pcie_bandwidth);
    bool rt_queues_task_empty();
};


} // namespace server
} // namespace missilebase