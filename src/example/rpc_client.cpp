#include "client/client.h"

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr << "Usage: " << std::string(argv[0]) 
                  << " port model_dir model_name [real_time] [num_sm_request] [num_sm_limit] [alloc_memory_size_kib] [perc_pcie_bandwidth] [estimated_runtime_ms] [slo_ms]\n";
        std::cerr << "Example: " << std::string(argv[0]) << " missile/resource/resnet18 resnet18 1\n";
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
    int estimated_runtime_ms = std::atoi(argv[9]);
    int slo_ms = std::atoi(argv[10]);


    missilebase::client::MISSILEClient client(std::string(DEFAULT_MISSILE_ADDR)+std::string(":")+std::to_string(port));
    // whether this client send real-time requests?
    ASSERT(client.init(real_time, perc_sm_request, perc_sm_limit, alloc_memory_size_kib, perc_pcie_bandwidth)); 
    
    std::cout << "loading '" << model_name << "' from " << "'"<< model_dir << "'\n";
    auto model = client.load_model(model_dir, model_name, estimated_runtime_ms, slo_ms);
    ASSERT(model.get() != nullptr);

    // Get or set the input/output data.
    // auto input_blob = model->get_input_blob();
    // model->load_input();
    // auto output_blob = model->get_output_blob();
    // auto output = model->get_output();

    std::cout << "submit inference request\n";
    
    auto task = model->infer(); // submit an inference request
    std::cout << "inference latency: " << std::chrono::duration_cast<std::chrono::microseconds>
                                          (task.finish - task.submit).count() / 1000.0 << " ms\n";

    return 0;
}