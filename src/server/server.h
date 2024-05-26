#pragma once

#include "util/common.h"
#include "util/shared_memory.h"
#include "rpc/missile.grpc.pb.h"
#include "server/base_scheduler.h"
#include <grpcpp/grpcpp.h>

namespace missilebase {
namespace server {


class MISSILEServer final : public missile::rpc::MISSILEService::Service {
public:
    MISSILEServer(const SchedulerType& type, const std::string& addr, const uint32_t cfs_period);
    virtual ~MISSILEServer() {}
    void run();

    void wait();

    void shutdown();
    
    BaseScheduler* get_scheduler() const {
        return scheduler.get();
    }
    
private: 
    // RPC handles
    grpc::Status SetPriority(
        grpc::ServerContext *context,
        const missile::rpc::SetPriorityRequest *request,
        missile::rpc::SetPriorityReply *reply
    ) override;

    grpc::Status LoadModel(
        grpc::ServerContext *context,
        const missile::rpc::LoadModelRequest *request,
        missile::rpc::LoadModelReply *reply
    ) override;

    grpc::Status RegisterBlob(
        grpc::ServerContext *context,
        const missile::rpc::RegisterBlobRequest *request,
        missile::rpc::RegisterBlobReply *reply
    ) override;
    
    grpc::Status GetBlob(
        grpc::ServerContext *context,
        const missile::rpc::GetBlobRequest *request,
        missile::rpc::GetBlobReply *reply
    ) override;

    grpc::Status SetBlob(
        grpc::ServerContext *context,
        const missile::rpc::SetBlobRequest *request,
        missile::rpc::SetBlobReply *reply
    ) override;

    grpc::Status Infer(
            grpc::ServerContext *context,
            const missile::rpc::InferRequest *request,
            missile::rpc::InferReply *reply
    ) override;

    grpc::Status Logout(
            grpc::ServerContext *context,
            const missile::rpc::LogoutRequest *request,
            missile::rpc::LogoutReply *reply
    ) override;

private:
    SchedulerType scheduler_type;
    std::string server_addr;
    std::unique_ptr<grpc::Server> rpc_server;
    std::unique_ptr<BaseScheduler> scheduler;
    std::mutex shm_mtx;
    struct SharedMemoryInfo {
        std::string name;
        std::shared_ptr<util::SharedMemory> shm;
        ModelID mid;
    };
    std::unordered_map<std::string, SharedMemoryInfo> shms;
};

} // namespace server 
} // namespace missilebase