//
// Created by Anonymous authors on 2024/1/11.
//

#pragma once

#include <future>

#include "threadsafe_queue.h"

namespace missilebase {
    namespace utils {
        struct GPULaunchKernelRequest {
            int qid;
            int op_offset;
            int kernel_offset;
            int TPC_request_num_cores;
            int TPC_limit_num_cores;
            double launch_progress;
            double runtime_ms;

            GPULaunchKernelRequest() {
                qid = 0;
                op_offset = 0;
                kernel_offset = 0;
                TPC_request_num_cores = 0;
                TPC_limit_num_cores = 0;
                launch_progress = 0;
            }

            GPULaunchKernelRequest(int _qid, int _op_offset, int _kernel_offset,
                                   int _TPC_request_num_cores, int _TPC_limit_num_cores,
                                   double _launch_progress, double _runtime_ms) {
                qid = _qid;
                op_offset = _op_offset;
                kernel_offset = _kernel_offset;
                TPC_request_num_cores = _TPC_request_num_cores;
                TPC_limit_num_cores = _TPC_limit_num_cores;
                launch_progress = _launch_progress;
                runtime_ms = _runtime_ms;
            }

            bool operator < (GPULaunchKernelRequest &b) const;
        };

        class GPULaunchKernelQueue : public ThreadSafeQueue<std::shared_ptr<GPULaunchKernelRequest>> {
        public:
            void wait_until_empty();

            void notify_empty();

        public:
            GPULaunchKernelQueue() {}
        };

        struct GPULaunchMemcpyRequest {
            void* src_ptr;
            void* dst_ptr;
            uint64_t copy_size_bytes;

            GPULaunchMemcpyRequest() {
                src_ptr = NULL;
                dst_ptr = NULL;
                copy_size_bytes = 0;
            }

            GPULaunchMemcpyRequest(void* _dst_ptr, void* _src_ptr, uint64_t _copy_size_bytes) {
                src_ptr = _src_ptr;
                dst_ptr = _dst_ptr;
                copy_size_bytes = _copy_size_bytes;
            }
        };

        class GPULaunchMemcpyQueue : public ThreadSafeQueue<std::shared_ptr<GPULaunchMemcpyRequest>> {
            public:
            void wait_until_empty();

            void notify_empty();

            public:
            GPULaunchMemcpyQueue();
        private:
            struct timespec ns_sleep;
        };
    }
}