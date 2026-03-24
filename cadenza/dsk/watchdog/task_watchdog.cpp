#include "../scheduler/dsk_api.h"
#include "../../ikm/ipc/ikm_bus.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>

namespace cadenza::dsk {

static constexpr int CONSECUTIVE_MISS_THRESHOLD = 3;

struct WatchdogAlert {
    uint32_t task_id;
    uint32_t consecutive_misses;
    uint64_t timestamp_us;
};

static std::atomic<bool> watchdog_running{false};
static std::thread watchdog_thread;

// External: access task registry deadline miss counters
// These are checked by the watchdog from the priority_queue module
extern std::atomic<int> task_count;

struct TaskSlot {
    CadenzaTask task;
    std::thread worker;
    std::atomic<bool> running;
    std::atomic<uint32_t> deadline_misses;
};
extern TaskSlot task_registry[];

static void watchdog_loop() {
    uint32_t watchdog_topic = ikm::ikm_topic_from_name("dsk/watchdog");

    while (watchdog_running.load(std::memory_order_acquire)) {
        int count = task_count.load(std::memory_order_acquire);

        for (int i = 0; i < count; i++) {
            if (!task_registry[i].task.active) continue;

            uint32_t misses = task_registry[i].deadline_misses.load(std::memory_order_acquire);
            if (misses >= CONSECUTIVE_MISS_THRESHOLD) {
                WatchdogAlert alert{};
                alert.task_id = task_registry[i].task.task_id;
                alert.consecutive_misses = misses;
                auto now = std::chrono::steady_clock::now();
                alert.timestamp_us = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch()).count());

                ikm::ikm_publish(watchdog_topic, &alert, sizeof(alert));
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int watchdog_start() {
    if (watchdog_running.load(std::memory_order_acquire)) return -1;

    watchdog_running.store(true, std::memory_order_release);
    watchdog_thread = std::thread(watchdog_loop);
    return 0;
}

void watchdog_stop() {
    watchdog_running.store(false, std::memory_order_release);
    if (watchdog_thread.joinable()) {
        watchdog_thread.join();
    }
}

} // namespace cadenza::dsk
