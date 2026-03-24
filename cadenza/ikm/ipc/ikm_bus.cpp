#include "ikm_bus.h"
#include "../shared_memory/shm_pool.h"
#include "../shared_memory/lockfree_queue.h"
#include <cstring>
#include <atomic>
#include <chrono>
#include <array>

namespace cadenza::ikm {

// Message reference stored in queue — points into shm, no data copy
struct ShmMessageRef {
    uint32_t topic_id;
    size_t shm_offset;
    size_t data_len;
    uint64_t timestamp_us;
};

struct TopicEntry {
    uint32_t topic_id;
    char name[64];
    SubscriberCallback subscribers[MAX_SUBSCRIBERS_PER_TOPIC];
    std::atomic<uint32_t> subscriber_count{0};
    bool active;
};

static TopicEntry topics[MAX_TOPICS];
static std::atomic<uint32_t> topic_count{0};
static std::atomic<bool> bus_initialized{false};

// Topic name hash table for O(1) lookup
static uint32_t hash_topic_name(const char* name) {
    uint32_t hash = 5381;
    while (*name) {
        hash = ((hash << 5) + hash) + static_cast<uint32_t>(*name++);
    }
    return hash % MAX_TOPICS;
}

int ikm_bus_init() {
    if (bus_initialized.load(std::memory_order_acquire)) return -1;

    std::memset(topics, 0, sizeof(topics));
    topic_count.store(0, std::memory_order_release);

    // Initialize shm pool (64MB default)
    int rc = shm_pool_init();
    if (rc != 0) return rc;

    bus_initialized.store(true, std::memory_order_release);
    return 0;
}

uint32_t ikm_topic_from_name(const char* name) {
    if (!name) return UINT32_MAX;

    uint32_t count = topic_count.load(std::memory_order_acquire);

    // Search existing topics
    for (uint32_t i = 0; i < count; i++) {
        if (topics[i].active && std::strcmp(topics[i].name, name) == 0)
            return topics[i].topic_id;
    }

    // Create new topic
    if (count >= MAX_TOPICS) return UINT32_MAX;

    uint32_t id = hash_topic_name(name);
    // Handle collision
    while (true) {
        bool found = false;
        for (uint32_t i = 0; i < count; i++) {
            if (topics[i].topic_id == id) {
                id = (id + 1) % MAX_TOPICS;
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    auto& entry = topics[count];
    entry.topic_id = id;
    std::strncpy(entry.name, name, sizeof(entry.name) - 1);
    entry.name[sizeof(entry.name) - 1] = '\0';
    entry.subscriber_count.store(0, std::memory_order_release);
    entry.active = true;

    topic_count.store(count + 1, std::memory_order_release);
    return id;
}

int ikm_publish(uint32_t topic_id, const void* data, size_t len) {
    if (!bus_initialized.load(std::memory_order_acquire)) return -1;

    // Data is already in shared memory — we pass the pointer directly to subscribers
    // No memcpy. The publisher writes to shm, subscribers read from the same address.

    uint32_t count = topic_count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topics[i].topic_id == topic_id && topics[i].active) {
            uint32_t sub_count = topics[i].subscriber_count.load(std::memory_order_acquire);
            for (uint32_t s = 0; s < sub_count; s++) {
                if (topics[i].subscribers[s]) {
                    topics[i].subscribers[s](topic_id, data, len);
                }
            }
            return 0;
        }
    }

    return -2; // Topic not found
}

int ikm_subscribe(uint32_t topic_id, SubscriberCallback callback) {
    if (!bus_initialized.load(std::memory_order_acquire)) return -1;
    if (!callback) return -2;

    uint32_t count = topic_count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topics[i].topic_id == topic_id && topics[i].active) {
            uint32_t sub_idx = topics[i].subscriber_count.load(std::memory_order_acquire);
            if (sub_idx >= MAX_SUBSCRIBERS_PER_TOPIC) return -3;

            topics[i].subscribers[sub_idx] = callback;
            topics[i].subscriber_count.store(sub_idx + 1, std::memory_order_release);
            return 0;
        }
    }

    return -4; // Topic not found
}

void ikm_bus_shutdown() {
    bus_initialized.store(false, std::memory_order_release);
    topic_count.store(0, std::memory_order_release);
    shm_pool_shutdown();
}

} // namespace cadenza::ikm
