#include "../util/buddy_allocator.h"
#include "memory_manager_base.h"

namespace missilebase {
    namespace executor {
        MemoryManagerBase::MemoryManagerBase(bool is_real_time, 
                                             uint64_t _capacity_size_bytes,
                                             uint64_t _alloc_granularity_bytes,
                                             std::shared_ptr<BuddyAllocator> rt_buddy_allocator,
                                             std::shared_ptr<BuddyAllocator> be_buddy_allocator) {
            if(is_real_time) {
                this->buddy_allocator = rt_buddy_allocator;
            } else {
                this->buddy_allocator = be_buddy_allocator;
            }
            capacity_size_bytes = _capacity_size_bytes;
            allocated_size_bytes.store(0);
        }

        uint64_t MemoryManagerBase::get_capacity_size_bytes() {
            return capacity_size_bytes;
        }

        uint64_t MemoryManagerBase::get_free_size_bytes() {
            return capacity_size_bytes - allocated_size_bytes.load();
        }

        uint64_t MemoryManagerBase::get_allocated_size_bytes() {
            return allocated_size_bytes.load();
        }
    }
}