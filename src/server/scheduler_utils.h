//
// Created by Anonymous authors on 2023/9/19.
//
#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "util/threadsafe_queue.h"
#include "executor/executor_base.h"
#include "executor/naive/naive_executor.h"
#include "executor/cuda/cuda_executor.h"
#include "executor/temporal_multiplexing/temporal_multiplexing_executor.h"
#include "executor/spatial_multiplexing/spatial_multiplexing_executor.h"
#include "executor/mpsplus_multiplexing/mpsplus_executor.h"

namespace missilebase {
    namespace server {

        typedef uint32_t ModelID;
        typedef uint32_t QueueID;
        typedef uint32_t TaskID;

        enum SchedulerType {
            Missile,
            MissileTemporal,
            TemporalMultiplexing,
            Clockwork,
            MultiTask,
            Orion,
            MPSPlus,
            Naive,
            KernelLatencyRecorder
        };

        enum ScheduleMode {
            Default, // the naive schedule for only one model
            ClockworkPolicy, // Clockwork's policy for temporal multiplexing (OSDI'20)
            ReefPolicy, // Reef's policy for spatial multiplexing (OSDI'22)
            MultiTaskPolicy, // Use libsmctrl to divide compute units.
            OrionPolicy, // Orion's policy for spatial multiplexing (EuroSys'24)
            MPSPlusPolicy,
            NoPreempt, // no preemption
            MultiStream, // multiple GPU streams
            WaitPreempt, // wait-based preemption
            MISSILE,
            Reset, // reset-based preemption without DKP
            MissileSpatialPolicy,
            MissileTemporalPolicy,
        };

        enum TaskQueueType {
            RealTimeQueue,
            BestEffortQueue,
        };

        enum TaskState {
            Init,
            Waiting,
            Executing,
            Reject, // Only for Clockwork policy
            Finish
        };

        struct TaskAbstractModel {
            std::shared_ptr<missilebase::executor::ExecutorBase> executor;
            QueueID qid;
            bool is_real_time;
            int estimated_runtime_ms, slo_ms;

            TaskAbstractModel(SchedulerType scheduler_type) {
                if(scheduler_type == Missile || scheduler_type == MissileTemporal) {
                    executor = std::make_shared<missilebase::executor::CUDAExecutor>();
                } else if(scheduler_type == TemporalMultiplexing || scheduler_type == Clockwork) {
                    executor = std::make_shared<missilebase::executor::TemporalMultiplexingExecutor>();
                } else if(scheduler_type == Orion || scheduler_type == MultiTask) {
                    executor = std::make_shared<missilebase::executor::SpatialMultiplexingExecutor>();
                } else if(scheduler_type == MPSPlus) {
                    executor = std::make_shared<missilebase::executor::MPSPlusExecutor>();
                } else if(scheduler_type == Naive || scheduler_type == KernelLatencyRecorder) {
                    executor = std::make_shared<missilebase::executor::NaiveExecutor>();
                } else {
                    LOG(ERROR) << "Unsupported scheduler type: " << scheduler_type;
                }
            }
        };

        struct Task {
        public:
            std::shared_ptr <TaskAbstractModel> model;
            QueueID qid;
            TaskID id;
            volatile TaskState state;
            int create_step_offset;
            int swap_in_step_offset;
            int swap_out_step_offset;
            int launch_step_offset;
            int reclaim_step_offset;
            std::mutex mtx;
            std::condition_variable cv;
            std::chrono::system_clock::time_point submit; // when this task is created
            std::chrono::system_clock::time_point start; // when this task is scheduled
            std::chrono::system_clock::time_point end; // when this task is completed
            bool preempted;
            bool padding;
            bool padding_to_finish;

            Task() {
                create_step_offset = 0;
                swap_in_step_offset = 0;
                swap_out_step_offset = 0;
                launch_step_offset = 0;
                reclaim_step_offset = 0;
                preempted = 0;
            }
        public:
            bool is_preempted() const;

            bool is_finished() const;

            bool is_padded() const;

            bool is_padded_to_complete() const;

            std::vector <std::chrono::system_clock::time_point> get_timestamp() const;
        };

        struct TaskQueue {
            ThreadSafeQueue<std::shared_ptr<Task>> task_queue;
            executor::GPUStream_t stream;
            int TPC_request_num_cores, TPC_limit_num_cores, alloc_memory_size_kib, perc_pcie_bandwidth;
        };

        class CPUCoreManager {
        public:
            std::atomic<int> curr_cpu_core_id;
            void init(int num_cpu_cores);
            Status assign_one_cpu_core(int &cpu_id);
            Status pin_given_cpu_core(int cpu_id);
            Status pin_given_cpu_core_list(std::vector<int> &pin_core_id_list);
            void pin_one_cpu_core();
            void pin_multiple_cpu_core(int num_cores);
        private:
            int num_cpu_cores;
        };
    }
}
