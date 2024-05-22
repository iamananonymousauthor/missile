#pragma once

#include <atomic>
#include <mutex>
#include <assert.h>
#include <vector>
#include <future>

#include "common.h"

namespace missilebase {
// TODO: replace it with a lock-free queue
template<typename Element>
class ThreadSafeQueue {
public:
    std::promise<bool> prom;
    std::shared_future<bool> future;

    enum { Capacity = 1000000 };

    ThreadSafeQueue() : _tail(0), _head(0), future(prom.get_future()) {
        _array.resize(Capacity);
        num_elements.store(0);
    }   

    virtual ~ThreadSafeQueue() {
    }

    ThreadSafeQueue(const ThreadSafeQueue &queue) = delete;

    ThreadSafeQueue(ThreadSafeQueue && queue) noexcept {
        _tail.store(queue._tail.load());
        _head.store(queue._head.load());
        _array = std::move(queue._array);
    } 

    /* Producer only: updates tail index after setting the element in place */
    bool push(const Element& item)
    {	
        // quick fix: lock the producers
        std::unique_lock<std::mutex> lock(mtx);
        auto current_tail = _tail.load();            
        auto next_tail = increment(current_tail);
        num_elements.fetch_add(1);
        if(next_tail != _head.load())                         
        {
            _array[current_tail] = item;               
            _tail.store(next_tail);                    
            return true;
        }
        
        return false;  // full queue
    }

    /* Consumer only: updates head index after retrieving the element */
    void pop()
    {
        std::unique_lock<std::mutex> lock(mtx);
        const auto current_head = _head.load();  
        ASSERT_MSG(current_head != _tail.load(), "ThreadSafeQueue: Empty queue!");   // empty queue
        _head.store(increment(current_head));
        num_elements.fetch_sub(1);
    }

    /* Consumer only: updates head index after retrieving the element */
    bool pop_and_notify_empty()
    {
        std::unique_lock<std::mutex> lock(mtx);
        const auto current_head = _head.load();
        ASSERT_MSG(current_head != _tail.load(), "ThreadSafeQueue: Empty queue!");   // empty queue
        _head.store(increment(current_head));
        num_elements.fetch_sub(1);
        if(num_elements.load() == 0) {
            prom.set_value(true);
            return true;
        }
        return false;
    }

    Element& front()
    {
        std::unique_lock<std::mutex> lock(mtx);
        const auto current_head = _head.load();
        ASSERT_MSG(current_head != _tail.load(), "ThreadSafeQueue: Empty queue!");   // empty queue
        auto &item = _array[current_head]; 
        return item;
    }

    bool empty() const {
        // std::unique_lock<std::mutex> lock(mtx);
        return (_head.load() == _tail.load());
    }

    bool full() const
    {
        const auto next_tail = increment(_tail.load());
        return (next_tail == _head.load());
    }

    uint32_t size() const {
        return num_elements.load();
    }
protected:
    std::mutex mtx;
    size_t increment(size_t idx) const
    {
        return (idx + 1) % Capacity;
    }
    std::atomic<size_t>  _tail;  
    std::vector<Element> _array;
    std::atomic<size_t>  _head;
    std::atomic<uint32_t> num_elements;
};
} // namespace missilebase