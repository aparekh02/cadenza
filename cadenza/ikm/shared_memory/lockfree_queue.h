#pragma once
#ifndef CADENZA_IKM_LOCKFREE_QUEUE_H
#define CADENZA_IKM_LOCKFREE_QUEUE_H

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace cadenza::ikm {

template<typename T, size_t Capacity>
class LockfreeQueue {
public:
    LockfreeQueue() : head_(0), tail_(0) {}

    bool push(const T* item);
    bool pop(T* item);
    bool empty() const;
    size_t size() const;

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    T buffer_[Capacity];
};

} // namespace cadenza::ikm

#endif
