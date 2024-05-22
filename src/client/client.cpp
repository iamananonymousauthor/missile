#include "client/client.h"
#include "util/shared_memory.h"

#include <grpcpp/grpcpp.h>

namespace missilebase {
namespace client {

MISSILEClient::MISSILEClient(const std::string &server_addr) : rpc_client(nullptr) {
    LOG(INFO) << "Create MISSILEClient to " << server_addr;

    rpc_client = MISSILEService::NewStub(
                    grpc::CreateChannel(server_addr, grpc::InsecureChannelCredentials())
                 );

    ASSERT_MSG(rpc_client.get() != nullptr, "cannot create rpc client");
    LOG(INFO) << "Create MISSILEClient succeeds";
}

bool MISSILEClient::init(bool real_time, int perc_sm_request,
                      int perc_sm_limit, int alloc_memory_size_kib, int perc_pcie_bandwidth) {
    // set client (task queue) priority
    grpc::ClientContext ctx;
    missilebase::rpc::SetPriorityRequest request;
    missilebase::rpc::SetPriorityReply reply;
    request.set_rt(real_time);
    request.set_perc_sm_request(perc_sm_request);
    request.set_perc_sm_limit(perc_sm_limit);
    request.set_alloc_memory_size_kib(alloc_memory_size_kib);
    request.set_perc_pcie_bandwidth(perc_pcie_bandwidth);
    auto status = rpc_client->SetPriority(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());
    qid = reply.qid();
    this->real_time = real_time;
    return true;
}

std::shared_ptr<ModelHandle> MISSILEClient::load_model(
    const std::string& model_dir,
    const std::string& name,
    const int estimated_runtime_ms,
    const int slo_ms
) {
    grpc::ClientContext ctx;
    missilebase::rpc::LoadModelRequest request;
    missilebase::rpc::LoadModelReply reply;
    LOG(INFO) << "Loading model " << name;
    request.set_rt(real_time);
    request.set_dir(model_dir);
    request.set_name(name);
    request.set_qid(qid);
    request.set_estimated_runtime_ms(estimated_runtime_ms);
    request.set_slo_ms(slo_ms);
    auto status = rpc_client->LoadModel(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
    std::shared_ptr<ModelHandle> model = 
        std::make_shared<ModelHandle>(
            rpc_client, reply.mid(), model_dir, name
        );
    {
        std::unique_lock<std::mutex> lock(models_mtx);
        models.push_back(model);
    }
    return model;
}

ModelHandle::ModelHandle(
    const std::shared_ptr<MISSILEService::Stub>& _rpc_client,
    int32_t _mid,
    const std::string& _dir, 
    const std::string& _name
) : rpc_client(_rpc_client), mid(_mid), dir(_dir), name(_name) {

}

// submit an inference task. wait for completion.
TaskHandle ModelHandle::infer() {
    grpc::ClientContext ctx;
    missilebase::rpc::InferRequest request;
    missilebase::rpc::InferReply reply;
    request.set_mid(mid);
    TaskHandle t;
    t.submit = std::chrono::system_clock::now();
    auto status = rpc_client->Infer(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());
    t.finish = std::chrono::system_clock::now();
    t.tid = reply.tid();
    t.response = reply.response();
    return t;
}

// submit an asynchronous inference task.
TaskHandle ModelHandle::infer_async() {
    return TaskHandle();
}

// get the poniter of input shared memory.
std::shared_ptr<util::SharedMemory> ModelHandle::get_input_blob(const std::string& name) {
    if (input_blob.get() == nullptr) 
        input_blob = register_blob(name, input_blob_key);
    return input_blob;
}

// get the poniter of output shared memory.
std::shared_ptr<util::SharedMemory> ModelHandle::get_output_blob(const std::string& name) {
    if (output_blob.get() == nullptr) {
        output_blob = register_blob(name, output_blob_key);
    }
    return output_blob;
}

std::shared_ptr<util::SharedMemory> ModelHandle::register_blob(const std::string& name, std::string& key) {
    grpc::ClientContext ctx;
    missilebase::rpc::RegisterBlobRequest request;
    missilebase::rpc::RegisterBlobReply reply;

    request.set_mid(mid);
    request.set_name(name);
    auto status = rpc_client->RegisterBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
    
    std::shared_ptr<util::SharedMemory> shm =
        std::make_shared<util::SharedMemory>(reply.key(), reply.size());
    key = reply.key();
    return shm;
}

// load model input in MISSILE server. wait for completion.
void ModelHandle::load_input() {
    ASSERT(input_blob.get() != nullptr);
    grpc::ClientContext ctx;
    missilebase::rpc::SetBlobRequest request;
    missilebase::rpc::SetBlobReply reply;
    request.set_key(input_blob_key);
    auto status = rpc_client->SetBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
}

// load model output in MISSILE server. wait for completion.
void ModelHandle::get_output() {
    ASSERT(output_blob.get() != nullptr);
    grpc::ClientContext ctx;
    missilebase::rpc::GetBlobRequest request;
    missilebase::rpc::GetBlobReply reply;
    request.set_key(output_blob_key);
    auto status = rpc_client->GetBlob(&ctx, request, &reply);
    ASSERT_MSG(status.ok(), status.error_message());
    ASSERT(reply.succ());    
}

int32_t ModelHandle::get_mid() const {
    return this->mid;
}

// submit an inference task. wait for completion.
void ModelHandle::logout() {
        /*grpc::ClientContext ctx;
        missilebase::rpc::LogoutRequest request;
        missilebase::rpc::LogoutReply reply;
        request.set_mid(mid);
        TaskHandle t;
        t.submit = std::chrono::system_clock::now();
        auto status = rpc_client->Logout(&ctx, request, &reply);
        ASSERT_MSG(status.ok(), status.error_message());
        ASSERT(reply.succ());
        t.finish = std::chrono::system_clock::now();
        t.tid = reply.tid();
        return t;*/
    }

} // namespace client
} // namespace missilebase