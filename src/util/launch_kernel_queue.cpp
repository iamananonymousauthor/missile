//
// Created by Anonymous authors on 2024/1/11.
//

#include "launch_kernel_queue.h"

namespace missilebase {
    namespace utils {

        void GPULaunchKernelQueue::wait_until_empty() {
            future.get();
            prom = std::promise<bool>();
            future = std::shared_future<bool>(prom.get_future());
            return;
        }

        void GPULaunchKernelQueue::notify_empty() {
            prom.set_value(true);
            return;
        }

        bool GPULaunchKernelRequest::operator < (GPULaunchKernelRequest &b) const {
            return this->launch_progress < b.launch_progress;
        }

        GPULaunchMemcpyQueue::GPULaunchMemcpyQueue() {
            ns_sleep.tv_sec = 0;
            ns_sleep.tv_nsec = 1000;
            prom = std::promise<bool>();
            future = std::shared_future<bool>(prom.get_future());
        }

        void GPULaunchMemcpyQueue::wait_until_empty() {
            /*if(empty()) {
                return;
            }*/
            //auto start_time = std::chrono::system_clock::now();
            while(num_elements.load()>0) {
                /*auto now_time = std::chrono::system_clock::now();
                if(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()>1000000) {
                    LOG(ERROR) << "DEBUG: WAIT TIMEOUT, requests remained = %d" << num_elements.load();
                    //break;
                }*/
                nanosleep(&ns_sleep, 0);
            }
            //future.get();
            //prom = std::promise<bool>();
            //future = std::shared_future<bool>(prom.get_future());
            return;
        }

        void GPULaunchMemcpyQueue::notify_empty() {
            //LOG(INFO) << "DEBUG: Notify GPULaunchMemcpyQueue empty()!";
            //prom.set_value(true);
            return;
        }

    }
}