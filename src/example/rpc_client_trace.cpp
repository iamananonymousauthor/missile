#include "client/client.h"
#include <thread>
#include <fstream>
#include <queue>
#include <math.h>

#include "../util/threadsafe_queue.h"
#include "rpc_client_utils.h"

#define THOUSAND 1000.0

void func_task_submit(int duration_time_sec, std::vector<int> trace_ms, std::shared_ptr<TaskQueue> &task_queue){
    // make sure the real_time queue is created for convenience.
    int curr_trace_idx = 0;
    auto start_time = std::chrono::system_clock::now();
    double base_offset_ms=0;

    while (true) {
        auto now_time = std::chrono::system_clock::now();
        double curr_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count();
        if(duration_time_sec>0 && std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()>1000000*duration_time_sec) {
            break;
        }
        if(task_queue->task_queue.size()>1000) {
            continue;
        }
        if(curr_timestamp_us/1000.0 >= trace_ms[curr_trace_idx]+base_offset_ms) {
            std::shared_ptr<Task> submit_task = std::make_shared<Task>();
            submit_task->submit = std::chrono::system_clock::now();
            task_queue->task_queue.push(submit_task);
            curr_trace_idx++;
            if(curr_trace_idx>=trace_ms.size()) {
                curr_trace_idx = 0;
                base_offset_ms+=60000;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

int main(int argc, char** argv) {
    if (argc != 14) {
        std::cerr << "Usage: " << std::string(argv[0]) << " port model_dir model_name [real_time] [num_sm_request] [num_sm_limit] [alloc_memory_size_kib] [perc_pcie_bandwidth] [path_to_trace_file] [duration(s)] [estimated_runtime_ms] [slo_ms] [record_file_path]\n";
        std::cerr << "Example: " << std::string(argv[0]) << " missile/resource/resnet18 resnet18 1 10\n";
        return -1;
    }

    int port = std::atoi(argv[1]);
    std::string model_dir(argv[2]);
    std::string model_name(argv[3]);
    int real_time = std::atoi(argv[4]);
    int perc_sm_request = std::atoi(argv[5]);
    int perc_sm_limit = std::atoi(argv[6]);
    int alloc_memory_size_kib = std::atoi(argv[7]);
    int perc_pcie_bandwidth = std::atoi(argv[8]);
    std::string path_to_trace_file(argv[9]);
    int duration_time_sec = std::atoi(argv[10]);
    int estimated_runtime_ms = std::atoi(argv[11]);
    int slo_ms = std::atoi(argv[12]);
    std::string record_file_path(argv[13]);

    //assert(duration_time_sec <= 60);

    std::vector<int> trace_ms;
    FILE *fp = fopen(path_to_trace_file.c_str(), "r");
    assert(fp!=NULL);
    int launch_timestamp_ms;
    while(fscanf(fp, "%d", &launch_timestamp_ms) != EOF) {
        trace_ms.push_back(launch_timestamp_ms);
    }

    std::ofstream of(record_file_path, std::ios::out);
    std::shared_ptr<TaskQueue> task_queue = std::make_shared<TaskQueue>();

    missilebase::client::MISSILEClient client(std::string(DEFAULT_MISSILE_ADDR)+std::string(":")+std::to_string(port));
    ASSERT(client.init(real_time, perc_sm_request, perc_sm_limit, alloc_memory_size_kib, perc_pcie_bandwidth)); // whether this client send real-time requests?
    
    std::cout << "loading '" << model_name << "' from " << "'"<< model_dir << "'\n";
    auto model = client.load_model(model_dir, model_name, estimated_runtime_ms, slo_ms);
    ASSERT(model.get() != nullptr);

    // Get or set the input/output data.
    // auto input_blob = model->get_input_blob();
    // model->load_input();
    // auto output_blob = model->get_output_blob();
    // auto output = model->get_output();

    std::cout << "submit inference requests\n";
    std::unique_ptr<std::thread> task_submit_thread, task_execute_thread;
    task_submit_thread.reset(new std::thread(func_task_submit, duration_time_sec, std::ref(trace_ms), 
                                             std::ref(task_queue)));

    task_execute_thread.reset(new std::thread(func_task_execute, std::ref(model), 
                                              duration_time_sec, 
                                              std::ref(of), std::ref(task_queue)));
    task_submit_thread->join();
    task_execute_thread->join();
    return 0;
}