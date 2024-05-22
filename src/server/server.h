#pragma once

#include "util/common.h"
#include "util/shared_memory.h"
#include "rpc/missile.grpc.pb.h"
#include "server/base_scheduler.h"
#include <grpcpp/grpcpp.h>

namespace missilebase {
namespace server {


class MISSILEServer final : public missilebase::rpc::MISSILEService::Service {
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
        const missilebase::rpc::SetPriorityRequest *request,
        missilebase::rpc::SetPriorityReply *reply
    ) override;

    grpc::Status LoadModel(
        grpc::ServerContext *context,
        const missilebase::rpc::LoadModelRequest *request,
        missilebase::rpc::LoadModelReply *reply
    ) override;

    grpc::Status RegisterBlob(
        grpc::ServerContext *context,
        const missilebase::rpc::RegisterBlobRequest *request,
        missilebase::rpc::RegisterBlobReply *reply
    ) override;
    
    grpc::Status GetBlob(
        grpc::ServerContext *context,
        const missilebase::rpc::GetBlobRequest *request,
        missilebase::rpc::GetBlobReply *reply
    ) override;

    grpc::Status SetBlob(
        grpc::ServerContext *context,
        const missilebase::rpc::SetBlobRequest *request,
        missilebase::rpc::SetBlobReply *reply
    ) override;

    grpc::Status Infer(
            grpc::ServerContext *context,
            const missilebase::rpc::InferRequest *request,
            missilebase::rpc::InferReply *reply
    ) override;

    grpc::Status Logout(
            grpc::ServerContext *context,
            const missilebase::rpc::LogoutRequest *request,
            missilebase::rpc::LogoutReply *reply
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