#pragma once

#include <atomic>

#include "../util/buddy_allocator.h"

namespace missilebase {
    namespace executor {
        using namespace missilebase::utils;

        class MemoryManagerBase {
        public:
            MemoryManagerBase(bool is_real_time, 
                              uint64_t capacity_size_bytes, 
                              uint64_t _alloc_granularity_bytes,
                              std::shared_ptr<BuddyAllocator> _rt_buddy_allocator,
                              std::shared_ptr<BuddyAllocator> _be_buddy_allocator);
            uint64_t get_capacity_size_bytes();
            uint64_t get_free_size_bytes();
            uint64_t get_allocated_size_bytes();

        protected:
            std::shared_ptr<BuddyAllocator> buddy_allocator;
            uint64_t capacity_size_bytes;
            std::atomic<uint64_t> allocated_size_bytes;
        };
    }
}