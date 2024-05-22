//
// Adapted from: https://github.com/wuwenbin/buddy2/tree/master
//

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "buddy_allocator.h"
#include "util/common.h"

#define LEFT_LEAF(index) ((index) * 2 + 1)
#define RIGHT_LEAF(index) ((index) * 2 + 2)
#define PARENT(index) ( ((index) + 1) / 2 - 1)

#define IS_POWER_OF_2(x) (!((x)&((x)-1)))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define ALLOC malloc
#define FREE free

namespace missilebase {
    namespace utils {
        static uint64_t fixsize(uint64_t size) {
            uint64_t pow = 0;
            while((1llu<<pow) <= size) {
                pow++;
            }
            pow--;
            return (1llu<<pow);
        }

        BuddyAllocator::BuddyAllocator(uint64_t _size_bytes, uint64_t _alloc_granularity_bytes) {
                actual_size_bytes = _size_bytes;
                assume_size_bytes = 1llu;
                if(actual_size_bytes % _alloc_granularity_bytes != 0) {
                    ASSERT_MSG(Status::InvalidArgument, "Buddy allocator: Invalid size " << actual_size_bytes << " bytes");
                }
                while(assume_size_bytes < actual_size_bytes) {
                    assume_size_bytes <<= 1llu;
                }
                longest = (uint64_t*) malloc((2llu * (assume_size_bytes / _alloc_granularity_bytes) + 1024)*sizeof(uint64_t));
                LOG(INFO) << "DEBUG: BuddyAllocator actual size_bytes = " << actual_size_bytes <<
                ", assume it to be " << assume_size_bytes << " bytes";
                alloc_granularity_bytes = _alloc_granularity_bytes;
                num_alloc_units = assume_size_bytes / _alloc_granularity_bytes;

                uint64_t node_size;
                int i;

                if (num_alloc_units < 1 || !IS_POWER_OF_2(num_alloc_units)) {
                    ASSERT_STATUS(Status::InvalidArgument);
                }

                node_size = num_alloc_units * 2;

                for (uint64_t i = 0; i < 2 * num_alloc_units - 1; ++i) {
                    if (IS_POWER_OF_2(i + 1))
                        node_size /= 2;
                    longest[i] = node_size;
                }

                uint64_t subtracted_num_alloc_units = (assume_size_bytes - actual_size_bytes) / alloc_granularity_bytes;
                uint64_t tmp = actual_size_bytes / alloc_granularity_bytes;
                uint64_t curr = 1llu<<25llu;
                std::vector<int> temp_alloced_offset_in_num_units;
                temp_alloced_offset_in_num_units.clear();
                while(tmp > 0) {
                    if(tmp >= curr) {
                        tmp -= curr;
                        uint64_t offset;
                        buddy_alloc_in_pow2(offset, curr);
                        temp_alloced_offset_in_num_units.push_back(offset);
                    }
                    curr >>= 1llu;
                }

                tmp = subtracted_num_alloc_units;
                curr = 1llu<<25llu;
                while(tmp > 0) {
                    if(tmp >= curr) {
                        tmp -= curr;
                        uint64_t offset;
                        buddy_alloc_in_pow2(offset, curr);
                    }
                    curr >>= 1llu;
                }

                for(auto &offset : temp_alloced_offset_in_num_units) {
                    buddy_free_in_pow2(offset);
                }
        }

        BuddyAllocator::~BuddyAllocator() {
            free(longest);
        }

        Status BuddyAllocator::buddy_alloc_in_pow2(uint64_t &offset, uint64_t size_in_num_units) {
            std::unique_lock<std::mutex> lock(buddy_mtx);
            uint64_t index = 0;
            uint64_t node_size;
            offset = 0;

            if (size_in_num_units <= 0) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            else if (!IS_POWER_OF_2(size_in_num_units)) {
                size_in_num_units = fixsize(size_in_num_units);
            }

            if (longest[index] < size_in_num_units)  {
                RETURN_STATUS(Status::InvalidArgument);
            }

            for (node_size = num_alloc_units; node_size != size_in_num_units; node_size /= 2) {
                if (longest[LEFT_LEAF(index)] >= size_in_num_units)
                    index = LEFT_LEAF(index);
                else
                    index = RIGHT_LEAF(index);
            }

            longest[index] = 0;
            offset = (index + 1) * node_size - num_alloc_units;

            while (index) {
                index = PARENT(index);
                longest[index] =
                        MAX(longest[LEFT_LEAF(index)], longest[RIGHT_LEAF(index)]);
            }
            return Status::Succ;
        }

        Status BuddyAllocator::buddy_free_in_pow2(uint64_t offset_in_num_units) {
            std::unique_lock<std::mutex> lock(buddy_mtx);
            uint64_t node_size, index = 0;
            uint64_t left_longest, right_longest;

            if(!(offset_in_num_units >= 0 && offset_in_num_units < num_alloc_units)) {
                RETURN_STATUS(Status::InvalidArgument);
            }

            node_size = 1;
            index = offset_in_num_units + num_alloc_units - 1;

            for (; longest[index]; index = PARENT(index)) {
                node_size *= 2;
                if (index == 0) {
                    RETURN_STATUS(Status::Fail);
                }
            }

            longest[index] = node_size;

            while (index) {
                index = PARENT(index);
                node_size *= 2;

                left_longest = longest[LEFT_LEAF(index)];
                right_longest = longest[RIGHT_LEAF(index)];

                if (left_longest + right_longest == node_size)
                    longest[index] = node_size;
                else
                    longest[index] = MAX(left_longest, right_longest);
            }
            return Status::Succ;
        }

        Status BuddyAllocator::buddy_alloc(std::shared_ptr<std::vector<BuddyInfo>>& offset_in_num_units_list, uint64_t alloc_size_bytes) {
            uint64_t aligned_alloc_size_bytes = align_up(alloc_size_bytes, alloc_granularity_bytes);
            uint64_t aligned_alloc_num_units = aligned_alloc_size_bytes / alloc_granularity_bytes;
            uint64_t remained_unalloced_size_num_units = aligned_alloc_num_units;
            uint64_t curr_try_alloc_size_num_units = 1llu<<25llu;
            offset_in_num_units_list = std::make_shared<std::vector<BuddyInfo>>();
            offset_in_num_units_list->clear();
            bool alloc_success = true;
            while(remained_unalloced_size_num_units > 0) {
                while(remained_unalloced_size_num_units >= curr_try_alloc_size_num_units) {
                    remained_unalloced_size_num_units -= curr_try_alloc_size_num_units;
                    uint64_t offset_in_units;
                    if(buddy_alloc_in_pow2(offset_in_units, curr_try_alloc_size_num_units) != Status::Succ) {
                        break;
                    }
                    offset_in_num_units_list->push_back(BuddyInfo(offset_in_units, curr_try_alloc_size_num_units));
                }
                curr_try_alloc_size_num_units >>= 1llu;
            }
            if(remained_unalloced_size_num_units > 0) {
                alloc_success = false;
            }
            // if failed to allocate, release all allocated tensors
            if(!alloc_success) {
                LOG(ERROR) << "Buddy allocator: Failed to allocate size " << alloc_size_bytes;
                for(auto &buddy_info : (*offset_in_num_units_list)) {
                    this->buddy_free_in_pow2(buddy_info.buddy_offset_in_num_units);
                }
                return Status::Fail;
            } else {
                return Status::Succ;
            }
        }

        Status BuddyAllocator::buddy_free(std::shared_ptr<std::vector<BuddyInfo>> &offset_in_num_units_list) {
            for(auto &offset_in_num_units : (*offset_in_num_units_list)) {
                RETURN_STATUS(buddy_free_in_pow2(offset_in_num_units.buddy_offset_in_num_units));
            }
            offset_in_num_units_list = nullptr;
        }

        Status BuddyAllocator::size_of(uint64_t offset_bytes, uint64_t &size_bytes) {
            if(offset_bytes % alloc_granularity_bytes != 0) {
                RETURN_STATUS(Status::InvalidArgument);
            }

            uint64_t offset_in_num_units = offset_bytes / alloc_granularity_bytes;

            uint64_t node_size, index = 0;

            if(!(offset_in_num_units >= 0 && offset_in_num_units < num_alloc_units)) {
                RETURN_STATUS(Status::InvalidArgument);
            }

            node_size = 1;
            for (index = offset_in_num_units + num_alloc_units - 1; longest[index]; index = PARENT(index)) {
                node_size *= 2;
            }

            size_bytes = node_size * alloc_granularity_bytes;
            return Status::Succ;
        }

        // TODO: For debugging only
        Status BuddyAllocator::dump_log() {
            char canvas[65];
            int i, j;
            uint64_t node_size, offset;

            if (num_alloc_units > 64) {
                std::cerr << "buddy2_dump: (struct buddy2*)self is too big to dump";
                RETURN_STATUS(Status::InvalidArgument);
            }

            memset(canvas, '_', sizeof(canvas));
            node_size = num_alloc_units * 2;

            for (i = 0; i < 2 * num_alloc_units - 1; ++i) {
                if (IS_POWER_OF_2(i + 1))
                    node_size /= 2;

                if (longest[i] == 0) {
                    if (i >= num_alloc_units - 1) {
                        canvas[i - num_alloc_units + 1] = '*';
                    } else if (longest[LEFT_LEAF(i)] && longest[RIGHT_LEAF(i)]) {
                        offset = (i + 1) * node_size - num_alloc_units;

                        for (j = offset; j < offset + node_size; ++j)
                            canvas[j] = '*';
                    }
                }
            }
            canvas[num_alloc_units] = '\0';
            puts(canvas);
            return Status::Succ;
        }
    }
}