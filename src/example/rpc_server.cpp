#include "server/server.h"

#include <string>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#define BOOST_STACKTRACE_LINK
#include <boost/stacktrace.hpp>
#include <iostream>

void my_handler(int sig) {
    std::cerr << boost::stacktrace::stacktrace();
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << std::string(argv[0]) << " scheduler_type cfs_period server_idx\n";
        std::cerr << "Example: " << std::string(argv[0]) << " missile 128 0\n";
        return -1;
    }
    signal(SIGINT, my_handler);
    signal(SIGTERM, my_handler);
    signal(SIGABRT, my_handler);
    signal(SIGSEGV, my_handler);

    std::string scheduler_type(argv[1]);
    uint32_t cfs_period = std::atoi(argv[2]);
    uint32_t server_idx = std::atoi(argv[3]);
    /*int mps_ratio_percentage = std::atoi(argv[2]);
    std::string env_mps_ratio_percentage = std::string("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=")+std::to_string(mps_ratio_percentage);
    if (putenv((char*)env_mps_ratio_percentage.c_str()) != 0) {
        std::cout << "putenv failed" << std::endl;
        exit(1);
    }*/
    if(scheduler_type.compare("missile") == 0) {
        std::cout << "Scheduler: Missile" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::Missile,
                                        std::string(DEFAULT_MISSILE_ADDR)+std::string(":")+
                                        std::to_string(DEFAULT_MISSILE_PORT+server_idx),
                                        cfs_period);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("missiletemporal") == 0) {
        std::cout << "Scheduler: Missile" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::MissileTemporal,
                                        std::string(DEFAULT_MISSILE_ADDR)+std::string(":")+
                                        std::to_string(DEFAULT_MISSILE_PORT+server_idx), cfs_period);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("temporal") == 0) {
        std::cout << "Scheduler: Temporal Multiplexing" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::TemporalMultiplexing,
                                        std::string(DEFAULT_MISSILE_ADDR)+std::string(":")+
                                        std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("clockwork") == 0) {
        std::cout << "Scheduler: Temporal Multiplexing with Clockwork's policy" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::Clockwork, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("orion") == 0) {
        std::cout << "Scheduler: Spatial Multiplexing with Orion's policy" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::Orion, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("mpsplus") == 0) {
        std::cout << "Scheduler: MPSPlus policy" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::MPSPlus, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("multitask") == 0) {
        std::cout << "Scheduler: Spatial Multiplexing with Multitask policy" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::MultiTask, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("naive") == 0) {
        std::cout << "Scheduler: Naive" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::Naive, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    } else if(scheduler_type.compare("kernel_latency_recorder") == 0) {
        std::cout << "Scheduler: Naive with Kernel Latency Recorder" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::KernelLatencyRecorder, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+server_idx), 128);
        server.run();
        server.wait();
    }
    return 0;
}