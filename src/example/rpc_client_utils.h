//
// Created by Anonymous authors on 2024/1/7.
//

#pragma once

struct Task {
    std::chrono::system_clock::time_point submit; // when this task is created
    std::chrono::system_clock::time_point start; // when this task is scheduled
    std::chrono::system_clock::time_point end; // when this task is completed
};

struct TaskQueue {
    missilebase::ThreadSafeQueue<std::shared_ptr<Task>> task_queue;
};

enum TaskState {
    Init,
    Waiting,
    Executing,
    Reject, // Only for Clockwork policy
    Finish
};

void func_task_execute(std::shared_ptr<missilebase::client::ModelHandle> &model, int duration_time_sec,
                       std::ofstream &of, 
                       std::shared_ptr<TaskQueue> &task_queue) {
    auto start_time = std::chrono::system_clock::now();
    int finish_task_num = 0;
    while (true) {
        auto now_time = std::chrono::system_clock::now();
        if (duration_time_sec > 0 &&
            std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count() >
            1000000 * duration_time_sec) {
            break;
        }
        if (task_queue->task_queue.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        auto retrieved_task = task_queue->task_queue.front();
        task_queue->task_queue.pop();
        auto task = model->infer(); // submit an inference request
        double inference_latency_ms = std::chrono::duration_cast<std::chrono::microseconds>
                                      (task.finish - task.submit).count() / 1000.0;
        double total_latency_ms = std::chrono::duration_cast<std::chrono::microseconds>
                                       (task.finish - retrieved_task->submit).count() / 1000.0;
        std::cout << "client " << model->get_mid() << " inference latency: "
                  << inference_latency_ms << " ms, total latency: " << total_latency_ms << " ms " << ((task.response == (int)TaskState::Reject)? "(Reject)":"(Accept)") << std::endl;
        if (duration_time_sec > 0 &&
            std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count() >
            1000000 * duration_time_sec) {
            break;
        }
        finish_task_num++;
        of <<  std::chrono::time_point_cast<std::chrono::milliseconds>(retrieved_task->submit).time_since_epoch().count() << ","
           <<  std::chrono::time_point_cast<std::chrono::milliseconds>(task.submit).time_since_epoch().count() << ","
           << std::chrono::time_point_cast<std::chrono::milliseconds>(task.finish).time_since_epoch().count() << ","
           << inference_latency_ms << "," << total_latency_ms << ","
           << ((task.response == (int)TaskState::Reject)? "Reject":"Accept") << std::endl;
    }
    of << std::flush;
    //of << (double) finish_task_num / (double) duration_time_sec << ",-1" << std::endl;
}