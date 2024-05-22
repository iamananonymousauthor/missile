#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

#include "naive/naive_scheduler.h"
#include "temporal_multiplexing/temporal_multiplexing_scheduler.h"
#include "spatial_multiplexing/spatial_multiplexing_scheduler.h"
#include "missile_temporal/missile_temporal_scheduler.h"
#include "mps_plus/mpsplus_scheduler.h"

#include "scheduler_utils.h"
#include "server/server.h"
#include "util/shared_memory.h"

using namespace missilebase::util;

namespace missilebase {
namespace server {

MISSILEServer::MISSILEServer(const SchedulerType& type, const std::string& addr, const uint32_t cfs_period)
    : server_addr(addr), rpc_server(nullptr)
{
    LOG(INFO) << "Init server: listen address = " << addr;
    scheduler_type = type;
    if(type == MissileTemporal) {
        scheduler.reset(new MissileTemporalScheduler(0, ScheduleMode::MissileTemporalPolicy, 1024, 1024, cfs_period, 10, 4));
    } else if(type == TemporalMultiplexing) {
        scheduler.reset(new TemporalMultiplexingScheduler(0, ScheduleMode::WaitPreempt, 1, 1));
    } else if(type == Clockwork) {
        scheduler.reset(new TemporalMultiplexingScheduler(0, ScheduleMode::ClockworkPolicy));
    } else if(type == Orion) {
        scheduler.reset(new SpatialMultiplexingScheduler(0, ScheduleMode::OrionPolicy, 10, 4));
    } else if(type == MPSPlus) {
        scheduler.reset(new MPSPlusScheduler(0, ScheduleMode::MPSPlusPolicy));
    } else if(type == MultiTask) {
        scheduler.reset(new SpatialMultiplexingScheduler(0, ScheduleMode::MultiTaskPolicy));
    } else if(type == Naive) {
        scheduler.reset(new NaiveScheduler(0, false));
    } else if(type == KernelLatencyRecorder) {
        scheduler.reset(new NaiveScheduler(0, true));
    }
}

void MISSILEServer::run() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);

    rpc_server = builder.BuildAndStart();
    scheduler->run();
}

void MISSILEServer::wait() {
    ASSERT(rpc_server.get() != nullptr);
    rpc_server->Wait();
}

void MISSILEServer::shutdown() {
    ASSERT(rpc_server.get() != nullptr); 
    rpc_server->Shutdown();
    scheduler->shutdown();
}

grpc::Status MISSILEServer::SetPriority(
    grpc::ServerContext *context,
    const missilebase::rpc::SetPriorityRequest *request,
    missilebase::rpc::SetPriorityReply *reply
) {
    LOG(INFO) << "new client, real_time: " << request->rt() << " / " << request->perc_sm_request() 
              << " / " << request->perc_sm_limit() << " / " << request->alloc_memory_size_kib() 
              << " / " << request->perc_pcie_bandwidth();
    // create queue
    QueueID qid;
    Status s = scheduler->create_queue(
        request->rt() ? 
            TaskQueueType::RealTimeQueue
            : TaskQueueType::BestEffortQueue,
        request->perc_sm_request(),
        request->perc_sm_limit(),
        request->alloc_memory_size_kib(),
        request->perc_pcie_bandwidth(),
        qid
    );
    if (s != Status::Succ)
        reply->set_succ(false);
    else {
        reply->set_succ(true);
        reply->set_qid(qid);
    }
    LOG(INFO) << "DEBUG: SetPriority done.";
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::LoadModel( grpc::ServerContext *context,
                                    const missilebase::rpc::LoadModelRequest *request,
                                    missilebase::rpc::LoadModelReply *reply) {
    LOG(INFO) << "load model: " << request->name() << ", qid: " << request->qid();
    std::string prefix = request->dir() + "/" + request->name();
    std::string param_file = prefix + ".param";
    if (access(param_file.c_str(), F_OK) == -1) {
        param_file = "";
        LOG(INFO) << request->name() << " no param file";
    }

    ModelID mid;
    std::string load_cubin_filename;
    if(scheduler_type == Missile || scheduler_type == MissileTemporal) {
        // TODO: Not implemented!
        if(request->rt()) {
            load_cubin_filename = prefix + ".naive.cubin";
        } else {
            load_cubin_filename = prefix + ".be.cubin";
        }
    } else {
        load_cubin_filename = prefix + ".cubin";
    }
    double alloc_memory_size_gib;
    std::string memory_tag;
    double memory_capacity_gib = scheduler->get_memory_capacity_gib(request->rt(), request->qid());
    //LOG(INFO) << "DEBUG: scheduler type: " << scheduler_type;
    if(scheduler_type == SchedulerType::KernelLatencyRecorder) {
        memory_tag = "json";
    } else if(memory_capacity_gib <= 0.125) {
        memory_tag = "128mb.json";
    } else if(memory_capacity_gib <= 0.2) {
        memory_tag = "256mb.json";
    } else if(memory_capacity_gib <= 0.5) {
        memory_tag = "512mb.json";
    } else if(memory_capacity_gib <= 1) {
        memory_tag = "1gb.json";
    } else if(memory_capacity_gib <= 2) {
        memory_tag = "2gb.json";
    } else if(memory_capacity_gib <= 3) {
        memory_tag = "3gb.json";
    } else if(memory_capacity_gib <= 4) {
        memory_tag = "4gb.json";
    } else if(memory_capacity_gib <= 5) {
        memory_tag = "5gb.json";
    } else if(memory_capacity_gib <= 6) {
        memory_tag = "6gb.json";
    } else if(memory_capacity_gib <= 7) {
        memory_tag = "7gb.json";
    } else if(memory_capacity_gib <= 8) {
        memory_tag = "8gb.json";
    } else if(memory_capacity_gib <= 9) {
        memory_tag = "9gb.json";
    } else if(memory_capacity_gib <= 10) {
        memory_tag = "10gb.json";
    } else {
        ASSERT_MSG(false, "Invalid memory capacity: " << memory_capacity_gib << " GiB");
    }


    Status s = scheduler->load_model_to_host(
        scheduler_type,
        load_cubin_filename,
        prefix + ".execution." + memory_tag,
        param_file, // TODO: load param
        mid,
        request->rt(),
        request->qid(),
        request->estimated_runtime_ms(),
        request->slo_ms()
    );
    if (s != Status::Succ) {
        reply->set_succ(false);
        return grpc::Status::OK;
    } else {
        reply->set_mid(mid);
    }
    LOG(INFO) << "DEBUG: bind_model_queue to " << request->rt() ?
    "TaskQueueType::RealTimeQueue"
                  : "TaskQueueType::BestEffortQueue";
    LOG(INFO) << "DEBUG: model's executon runtime = " << request->estimated_runtime_ms() 
              << " ms, model's slo = " << request->slo_ms() << " ms";
    s = scheduler->bind_model_queue(
            request->rt() ?
            TaskQueueType::RealTimeQueue
                          : TaskQueueType::BestEffortQueue,
            request->qid(),
            mid);

    if (s != Status::Succ) {
        reply->set_succ(false); // TODO: unload model
        return grpc::Status::OK;
    }
    if (s != Status::Succ) {
        reply->set_succ(false); // TODO: unload model
        return grpc::Status::OK;
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::RegisterBlob(
    grpc::ServerContext *context,
    const missilebase::rpc::RegisterBlobRequest *request,
    missilebase::rpc::RegisterBlobReply *reply
) {
    reply->set_succ(false);
    size_t size;
    auto s = scheduler->get_data_size(request->mid(), request->name(), size);
    if (s != Status::Succ) return grpc::Status::OK;
    std::string key = std::to_string(request->mid()) + "_" + request->name();
    reply->set_key(key);
    reply->set_size(size);
    reply->set_succ(true);
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(key);
        if (iter == shms.end()) {
            auto shm = std::make_shared<util::SharedMemory>(key, size, true);
            SharedMemoryInfo shminfo;
            shminfo.name = request->name();
            shminfo.mid = request->mid();
            shminfo.shm = shm;
            shms.insert({key, shminfo});
        }
    }
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::GetBlob(
    grpc::ServerContext *context,
    const missilebase::rpc::GetBlobRequest *request,
    missilebase::rpc::GetBlobReply *reply
) {
    SharedMemoryInfo shminfo;
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(request->key());
        if (iter == shms.end()) {
            reply->set_succ(false);
            return grpc::Status::OK;
        }
        shminfo = iter->second;
    }
    auto s = scheduler->get_output(shminfo.mid, shminfo.shm->data(), shminfo.shm->size(), shminfo.name);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::SetBlob(
    grpc::ServerContext *context,
    const missilebase::rpc::SetBlobRequest *request,
    missilebase::rpc::SetBlobReply *reply
) {
    SharedMemoryInfo shminfo;
    {
        std::unique_lock<std::mutex> lock(shm_mtx);
        auto iter = shms.find(request->key());
        if (iter == shms.end()) {
            reply->set_succ(false);
            return grpc::Status::OK;
        }
        shminfo = iter->second;
    }
    auto s = scheduler->set_input(shminfo.mid, shminfo.shm->data(), shminfo.shm->size(), shminfo.name);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::Infer(
    grpc::ServerContext *context,
    const missilebase::rpc::InferRequest *request,
    missilebase::rpc::InferReply *reply
) {
    TaskID tid;
    auto s = scheduler->new_task(request->mid(), tid);
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        TaskState response;
        s = scheduler->wait_task(tid, response);
        reply->set_succ(true);
        if (s != Status::Succ)
            reply->set_succ(false);
        reply->set_tid(tid);
        reply->set_response(response);
    }
    return grpc::Status::OK;
}

grpc::Status MISSILEServer::Logout(
        grpc::ServerContext *context,
        const missilebase::rpc::LogoutRequest *request,
        missilebase::rpc::LogoutReply *reply
) {
    TaskID tid;
    auto s = scheduler->logout_model(request->mid());
    if (s != Status::Succ) {
        reply->set_succ(false);
    } else {
        reply->set_succ(true);
    }
    return grpc::Status::OK;
}

} // namespace server
} // namespace missilebase