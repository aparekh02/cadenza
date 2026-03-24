// Template implementations for LockfreeQueue are header-only.
// This file exists for the build system and contains explicit instantiations.

#include "lockfree_queue.h"
#include <cstring>

namespace cadenza::ikm {

// LockfreeQueue is SPSC (Single Producer, Single Consumer) lock-free ring buffer.
// Uses std::atomic only — no mutexes anywhere.
// Data is accessed via shared memory pointer — no memcpy for the data payload.

template<typename T, size_t Capacity>
bool LockfreeQueue<T, Capacity>::push(const T* item) {
    size_t current_tail = tail_.load(std::memory_order_relaxed);
    size_t next_tail = (current_tail + 1) % Capacity;

    if (next_tail == head_.load(std::memory_order_acquire)) {
        return false; // Queue full
    }

    // Store the item (this is a pointer/offset, not a data copy)
    buffer_[current_tail] = *item;
    tail_.store(next_tail, std::memory_order_release);
    return true;
}

template<typename T, size_t Capacity>
bool LockfreeQueue<T, Capacity>::pop(T* item) {
    size_t current_head = head_.load(std::memory_order_relaxed);

    if (current_head == tail_.load(std::memory_order_acquire)) {
        return false; // Queue empty
    }

    *item = buffer_[current_head];
    head_.store((current_head + 1) % Capacity, std::memory_order_release);
    return true;
}

template<typename T, size_t Capacity>
bool LockfreeQueue<T, Capacity>::empty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
}

template<typename T, size_t Capacity>
size_t LockfreeQueue<T, Capacity>::size() const {
    size_t head = head_.load(std::memory_order_acquire);
    size_t tail = tail_.load(std::memory_order_acquire);
    return (tail >= head) ? (tail - head) : (Capacity - head + tail);
}

// IKM message reference — passes shared memory pointer, never copies data
struct ShmMessageRef {
    uint32_t topic_id;
    size_t shm_offset;  // Offset into shared memory pool — zero copy
    size_t data_len;
    uint64_t timestamp_us;
};

// Explicit instantiation for IKM bus message queue
template class LockfreeQueue<ShmMessageRef, 4096>;

} // namespace cadenza::ikm
