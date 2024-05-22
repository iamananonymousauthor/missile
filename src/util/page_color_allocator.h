//
// Created by Anonymous authors on 2024/1/11.
//

#pragma once

#include <mutex>
#include <vector>
#include <queue>

#include "common.h"

namespace missilebase {
    namespace utils {
        class PageColorAllocator {
        private:
            // In total, there are "num_colors" colors.
            // [0, rt_colors) will be allocated to RT tasks
            // available_colors are reserved for RT tasks, each task can allocate one or more rt colors.
            // be_colors are reserved for all BE tasks, they share all be colors.
            int num_colors, num_rt_colors;
            std::mutex mtx;
            std::queue<int> available_colors;
            std::vector<int> rt_colors, be_colors;

        public:
            void setup(int num_colors, int rt_colors);

            Status color_alloc(bool is_real_time, std::vector<int> &colors, int num_alloc_colors);
            Status color_alloc_fixed(bool is_real_time, std::vector<int> &colors, int num_alloc_colors);

            Status color_free(std::vector<int> &colors);
        };
    }
}