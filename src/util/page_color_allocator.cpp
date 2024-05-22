//
// Created by Anonymous authors on 2024/1/11.
//

#include "page_color_allocator.h"
#include "common.h"

namespace missilebase {
    namespace utils {
        void PageColorAllocator::setup(int _num_colors, int _num_rt_colors) {
            num_colors = _num_colors;
            num_rt_colors = _num_rt_colors;
            for (int i = 0; i < num_rt_colors; i++) {
                available_colors.push(i);
                rt_colors.push_back(i);
            }
            for (int i = num_rt_colors; i < num_colors; i++) {
                be_colors.push_back(i);
            }
        }

        Status PageColorAllocator::color_alloc(bool is_real_time, std::vector<int> &colors, int num_alloc_colors) {
            std::unique_lock <std::mutex> lock(mtx);
            if (is_real_time) {
                if (available_colors.size() < num_alloc_colors) {
                    RETURN_STATUS(Status::Fail);
                }
                colors.resize(num_alloc_colors);
                for (int i = 0; i < num_alloc_colors; i++) {
                    colors[i] = available_colors.front();
                    available_colors.pop();
                }
            } else {
                LOG(INFO) << "DEBUG: allocate be colors #" << be_colors.size();
                for (auto &color: be_colors) {
                    colors.push_back(color);
                }
            }
            return Status::Succ;
        }

        Status PageColorAllocator::color_alloc_fixed(bool is_real_time, std::vector<int> &colors, int num_alloc_colors) {
            std::unique_lock <std::mutex> lock(mtx);
            if (is_real_time) {
                LOG(INFO) << "DEBUG: allocate rt colors #" << rt_colors.size();
                for (auto &color: rt_colors) {
                    colors.push_back(color);
                }
            } else {
                LOG(INFO) << "DEBUG: allocate be colors #" << be_colors.size();
                for (auto &color: be_colors) {
                    colors.push_back(color);
                }
            }
            return Status::Succ;
        }

        Status PageColorAllocator::color_free(std::vector<int> &colors) {
            std::unique_lock <std::mutex> lock(mtx);
            if (available_colors.size() + colors.size() > num_colors) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            for (int i = 0; i < colors.size(); i++) {
                available_colors.push(colors[i]);
            }
            return Status::Succ;
        }
    }
}