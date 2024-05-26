#pragma once

#include "rpc/missile.grpc.pb.h"
#include "util/common.h"
#include "util/shared_memory.h"

#include <glog/logging.h>

#include <string>
#include <memory>
#include <vector>

namespace missilebase {
namespace client {

using missile::rpc::MISSILEService;

class TaskHandle {
public:
    int32_t tid;
    std::chrono::system_clock::time_point submit, finish;
    int32_t response;
};

// ModelHandle can be used to submit inference task.
class ModelHandle {
public:
    ModelHandle(
        const std::shared_ptr<MISSILEService::Stub>& rpc_client,
        int32_t _mid, 
        const std::string& dir,
        const std::string& name
    );
    // submit an inference task. wait for completion.
    TaskHandle infer();

    // submit an asynchronous inference task.
    TaskHandle infer_async();

    // get the poniter of input shared memory.
    std::shared_ptr<util::SharedMemory> get_input_blob(const std::string& name = "data");

    // get the poniter of output shared memory.
    std::shared_ptr<util::SharedMemory> get_output_blob(const std::string& name = "output");

    // load model input in MISSILE server. wait for completion.
    void load_input();

    // load model output in MISSILE server. wait for completion.
    void get_output();

    // TODO: FIXME
    void get_blob();
    void set_blob();

    int32_t get_mid() const;

    void logout();
private:
    std::shared_ptr<MISSILEService::Stub> rpc_client;
    int32_t mid;
    std::string dir;
    std::string name;
    std::string input_blob_key, output_blob_key;
    std::shared_ptr<util::SharedMemory> input_blob;
    std::shared_ptr<util::SharedMemory> output_blob;

private:
    std::shared_ptr<util::SharedMemory> register_blob(const std::string& name, std::string& key);
    
};


// MISSILEClient is used to estabilish connection to MISSILE server
// and load models into the server.
class MISSILEClient {
public:
    MISSILEClient(const std::string &server_addr);
    // initialize the client
    // Each client should be configured with a priority.
    // The real-time clients will share a RT task queue.
    // Each best-effort client will have its own BE task queue.
    bool init(bool real_time = false, int perc_sm_request = 0, int perc_sm_limit = 0, 
              int alloc_memory_size_kib = 0, int perc_pcie_bandwidth = 0);

    // load a DNN model (in MISSILE server).
    std::shared_ptr<ModelHandle> load_model(
            const std::string& model_dir,
            const std::string& name,
            const int estimated_runtime_ms,
            const int slo_ms
    );

private:
    std::shared_ptr<MISSILEService::Stub> rpc_client;
    std::mutex models_mtx;
    std::vector<std::shared_ptr<ModelHandle>> models;
    int32_t qid;
    bool real_time;
};


} // namespace client
} // namespace missilebase