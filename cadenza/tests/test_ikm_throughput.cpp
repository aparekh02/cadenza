// Test 3: 16 publishers, 10,000 messages each, 16 topics concurrently.
// Verify zero dropped messages.

#include "../ikm/ipc/ikm_bus.h"
#include "../ikm/shared_memory/shm_pool.h"
#include <cassert>
#include <cstdio>
#include <atomic>
#include <thread>
#include <vector>

using namespace cadenza::ikm;

static constexpr int NUM_PUBLISHERS = 16;
static constexpr int MSGS_PER_PUBLISHER = 10000;

static std::atomic<int> received_counts[NUM_PUBLISHERS];

static void subscriber_callback(uint32_t topic_id, const void* data, size_t len) {
    (void)len;
    if (data) {
        int pub_idx = *static_cast<const int*>(data);
        if (pub_idx >= 0 && pub_idx < NUM_PUBLISHERS) {
            received_counts[pub_idx].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

int main() {
    std::printf("test_ikm_throughput:\n");

    // Init
    int rc = ikm_bus_init();
    assert(rc == 0 && "ikm_bus_init failed");

    // Create topics and subscribe
    uint32_t topic_ids[NUM_PUBLISHERS];
    for (int i = 0; i < NUM_PUBLISHERS; i++) {
        char name[32];
        std::snprintf(name, sizeof(name), "test/topic_%d", i);
        topic_ids[i] = ikm_topic_from_name(name);
        assert(topic_ids[i] != UINT32_MAX && "topic creation failed");

        received_counts[i].store(0, std::memory_order_release);
        rc = ikm_subscribe(topic_ids[i], subscriber_callback);
        assert(rc == 0 && "subscribe failed");
    }

    // Launch publishers
    std::vector<std::thread> publishers;
    for (int i = 0; i < NUM_PUBLISHERS; i++) {
        publishers.emplace_back([i, &topic_ids]() {
            for (int m = 0; m < MSGS_PER_PUBLISHER; m++) {
                int payload = i;
                ikm_publish(topic_ids[i], &payload, sizeof(payload));
            }
        });
    }

    for (auto& t : publishers) t.join();

    // Verify zero dropped messages
    int total_received = 0;
    for (int i = 0; i < NUM_PUBLISHERS; i++) {
        int count = received_counts[i].load(std::memory_order_acquire);
        std::printf("  Topic %d: received %d/%d\n", i, count, MSGS_PER_PUBLISHER);
        assert(count == MSGS_PER_PUBLISHER && "Dropped messages detected!");
        total_received += count;
    }

    int expected = NUM_PUBLISHERS * MSGS_PER_PUBLISHER;
    assert(total_received == expected && "Total message count mismatch");
    std::printf("  Total: %d/%d messages delivered, zero drops\n", total_received, expected);

    ikm_bus_shutdown();
    std::printf("  ALL PASSED\n");
    return 0;
}
