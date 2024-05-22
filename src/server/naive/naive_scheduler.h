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
#include "../../executor/naive/naive_executor.h"
#include "../../cuda_implementation/cuda_impl.h"

namespace missilebase {
namespace server {

class NaiveScheduler : public BaseScheduler {
public:
    NaiveScheduler(int _gpu_id, bool _record_latency_per_kernel);
    ~NaiveScheduler();

    Status load_model_from_host_to_device(ModelID& mid);
    Status bind_model_queue(
        const TaskQueueType& type,
        const QueueID& qid,
        const ModelID& mid
    );
    Status run();
    Status shutdown();

private:
    std::unique_ptr<std::thread> task_thread;

    bool record_latency_per_kernel;

    void execute_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);

    void loop_body_thread(std::shared_ptr<TaskQueue>& tqueue);
};


} // namespace server
} // namespace missilebase