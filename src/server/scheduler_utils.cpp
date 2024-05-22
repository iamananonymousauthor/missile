//
// Created by Anonymous authors on 2023/9/19.
//

#include <vector>

#include "scheduler_utils.h"

namespace missilebase {
    namespace server {
        std::vector<std::chrono::system_clock::time_point>
        Task::get_timestamp() const {
            return std::vector<std::chrono::system_clock::time_point>({submit, start, end });
        }


        bool Task::is_preempted() const {
            return preempted;
        }

        bool Task::is_finished() const {
            return this->model->executor->step_graph->is_finished(
                    create_step_offset, swap_in_step_offset,
                    swap_out_step_offset, launch_step_offset,
                    reclaim_step_offset);
        }

        bool Task::is_padded() const {
            return padding;
        }

        bool Task::is_padded_to_complete() const {
            return padding_to_finish;
        }

        void CPUCoreManager::init(int _num_cpu_cores) {
            LOG(INFO) << "CPUCoreManager: Initialize " << _num_cpu_cores << " CPU cores";
            num_cpu_cores = _num_cpu_cores;
            std::atomic_init(&curr_cpu_core_id, 0);
        }

        Status CPUCoreManager::assign_one_cpu_core(int &pin_core_id) {
            pin_core_id = curr_cpu_core_id.load();
            ASSERT_MSG(pin_core_id < num_cpu_cores, "Failed to pin cpu core #" << pin_core_id 
                                << ", reason: exceeds the max limit (" << num_cpu_cores << ").");
            curr_cpu_core_id++;
            return Status::Succ;
        }

        Status CPUCoreManager::pin_given_cpu_core(int pin_core_id) {
            pthread_t self = pthread_self();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(pin_core_id, &cpuset);
            ASSERT_MSG(pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset) == 0,
                       "Failed to pin cpu core #" << pin_core_id);
            return Status::Succ;
        }

        Status CPUCoreManager::pin_given_cpu_core_list(std::vector<int> &pin_core_id_list) {
            pthread_t self = pthread_self();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for(int &pin_core_id:pin_core_id_list) {
                CPU_SET(pin_core_id, &cpuset);
            }
            ASSERT_MSG(pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset) == 0,
                       "Failed to pin cpu cores");
            return Status::Succ;
        }

        void CPUCoreManager::pin_one_cpu_core() {
            int pin_core_id;
            ASSERT_STATUS(assign_one_cpu_core(pin_core_id));
            ASSERT_STATUS(pin_given_cpu_core(pin_core_id));
        }

        void CPUCoreManager::pin_multiple_cpu_core(int num_cores) {
            std::vector<int> pin_core_id_list;
            for(int i=0;i<num_cores;i++) {
                int pin_core_id;
                ASSERT_STATUS(assign_one_cpu_core(pin_core_id));
                pin_core_id_list.push_back(pin_core_id);
            }
            ASSERT_STATUS(pin_given_cpu_core_list(pin_core_id_list));
        }
    }
}
