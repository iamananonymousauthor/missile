//
// Adapted from: https://github.com/wuwenbin/buddy2/tree/master
//
#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <mutex>

#include "util/common.h"

namespace missilebase {
    namespace utils {
        static uint64_t fixsize(uint64_t size);

        struct BuddyInfo {
            uint64_t buddy_offset_in_num_units;
            uint64_t buddy_size_in_num_units;

            BuddyInfo() {
                buddy_offset_in_num_units = 0;
                buddy_size_in_num_units = 0;
            }

            BuddyInfo(uint64_t _buddy_offset_in_num_units, uint64_t _buddy_size_in_num_units) {
                buddy_offset_in_num_units = _buddy_offset_in_num_units;
                buddy_size_in_num_units = _buddy_size_in_num_units;
            }
        };

        class BuddyAllocator {
        protected:
            uint64_t actual_size_bytes, assume_size_bytes;
            uint64_t alloc_granularity_bytes, num_alloc_units;
            uint64_t *longest;
            std::mutex buddy_mtx;

        public:
            BuddyAllocator(uint64_t size_bytes, uint64_t alloc_granularity_bytes);
            ~BuddyAllocator();

            Status buddy_alloc_in_pow2(uint64_t &offset, uint64_t size_bytes);

            Status buddy_alloc(std::shared_ptr<std::vector<BuddyInfo>> &offset_in_num_units_list, uint64_t alloc_size_bytes);

            Status buddy_free_in_pow2(uint64_t offset_bytes);

            Status buddy_free(std::shared_ptr<std::vector<BuddyInfo>> &offset_in_num_units_list);

            Status size_of(uint64_t offset, uint64_t &size_bytes);

            Status dump_log();
        };
    }
}