#include "dsk_api.h"
#include "../../ikm/ipc/ikm_bus.h"
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

namespace cadenza::dsk {

// Forward declarations
int edf_schedule_task(const CadenzaTask* task, pthread_t thread);
int rms_schedule_task(const CadenzaTask* task, pthread_t thread);
int set_core_affinity(const CadenzaTask* task, pthread_t thread);

static constexpr int MAX_TASKS = 128;

struct TaskSlot {
    CadenzaTask task;
    std::thread worker;
    std::atomic<bool> running{false};
    std::atomic<uint32_t> deadline_misses{0};
};

static TaskSlot task_registry[MAX_TASKS];
static std::atomic<int> task_count{0};
static std::atomic<bool> dsk_running{false};

// Idle monitoring
static std::atomic<uint64_t> idle_budget_us{0};
static std::atomic<uint64_t> last_idle_check_us{0};

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

static TaskSlot* find_task(uint32_t task_id) {
    int count = task_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (task_registry[i].task.task_id == task_id && task_registry[i].task.active)
            return &task_registry[i];
    }
    return nullptr;
}

static void task_worker(TaskSlot* slot) {
    auto& task = slot->task;

    // Apply scheduling policy
    if (task.task_class == TaskClass::SAFETY || task.task_class == TaskClass::CONTROL) {
        rms_schedule_task(&task, pthread_self());
    } else if (task.task_class == TaskClass::SENSOR || task.task_class == TaskClass::INFERENCE) {
        edf_schedule_task(&task, pthread_self());
    }
    // COMMS uses SCHED_OTHER (default)

    set_core_affinity(&task, pthread_self());

    while (slot->running.load(std::memory_order_acquire)) {
        auto start = now_us();

        if (task.entry) {
            task.entry();
        }

        auto elapsed = now_us() - start;

        // Check deadline
        if (elapsed > task.deadline_us) {
            slot->deadline_misses.fetch_add(1, std::memory_order_relaxed);
        } else {
            slot->deadline_misses.store(0, std::memory_order_relaxed);
        }

        // Sleep for remainder of period
        if (elapsed < task.period_us) {
            uint64_t sleep_us = task.period_us - elapsed;
            idle_budget_us.fetch_add(sleep_us, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
}

int dsk_init() {
    if (dsk_running.load(std::memory_order_acquire)) return -1;

    std::memset(task_registry, 0, sizeof(task_registry));
    task_count.store(0, std::memory_order_release);
    idle_budget_us.store(0, std::memory_order_release);
    last_idle_check_us.store(now_us(), std::memory_order_release);
    dsk_running.store(true, std::memory_order_release);

    return 0;
}

int dsk_register_task(CadenzaTask* task) {
    if (!dsk_running.load(std::memory_order_acquire)) return -1;
    if (!task) return -2;

    int idx = task_count.load(std::memory_order_acquire);
    if (idx >= MAX_TASKS) return -3;

    auto& slot = task_registry[idx];
    slot.task = *task;
    slot.task.active = true;
    slot.running.store(true, std::memory_order_release);
    slot.deadline_misses.store(0, std::memory_order_release);

    slot.worker = std::thread(task_worker, &slot);

    task_count.store(idx + 1, std::memory_order_release);
    return 0;
}

int dsk_cancel_task(uint32_t task_id) {
    auto* slot = find_task(task_id);
    if (!slot) return -1;

    slot->running.store(false, std::memory_order_release);
    slot->task.active = false;
    if (slot->worker.joinable()) {
        slot->worker.join();
    }
    return 0;
}

int dsk_reprioritize(uint32_t task_id, int new_priority) {
    auto* slot = find_task(task_id);
    if (!slot) return -1;

    // CRITICAL: Only INFERENCE and COMMS can be reprioritized
    if (slot->task.task_class == TaskClass::SAFETY ||
        slot->task.task_class == TaskClass::CONTROL) {
        return -2; // Rejected — SAFETY/CONTROL priorities are immutable
    }

    if (slot->task.task_class == TaskClass::SENSOR) {
        return -2; // Rejected — SENSOR priorities are immutable
    }

    slot->task.priority = new_priority;

#if defined(__linux__) && !defined(CADENZA_TEST_MODE)
    if (slot->task.task_class == TaskClass::COMMS) {
        // COMMS uses SCHED_OTHER — nice value only
        return 0;
    }
    // INFERENCE uses SCHED_DEADLINE — re-apply
    if (slot->worker.joinable()) {
        edf_schedule_task(&slot->task, slot->worker.native_handle());
    }
#endif

    return 0;
}

bool dsk_has_active_control_tasks() {
    int count = task_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (task_registry[i].task.active &&
            task_registry[i].running.load(std::memory_order_acquire) &&
            task_registry[i].task.task_class == TaskClass::CONTROL) {
            return true;
        }
    }
    return false;
}

// Called periodically to check idle budget and publish to IKM
void dsk_check_idle_budget() {
    uint64_t now = now_us();
    uint64_t last = last_idle_check_us.load(std::memory_order_acquire);
    uint64_t window_us = 100000; // 100ms

    if (now - last >= window_us) {
        uint64_t idle = idle_budget_us.exchange(0, std::memory_order_acq_rel);
        last_idle_check_us.store(now, std::memory_order_release);

        // If idle budget > 80ms in last 100ms, publish to IKM
        if (idle > 80000) {
            uint32_t topic = ikm::ikm_topic_from_name("dsk/idle");
            uint64_t idle_ms = idle / 1000;
            ikm::ikm_publish(topic, &idle_ms, sizeof(idle_ms));
        }
    }
}

void dsk_shutdown() {
    dsk_running.store(false, std::memory_order_release);

    int count = task_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        task_registry[i].running.store(false, std::memory_order_release);
        task_registry[i].task.active = false;
        if (task_registry[i].worker.joinable()) {
            task_registry[i].worker.join();
        }
    }
    task_count.store(0, std::memory_order_release);
}

} // namespace cadenza::dsk
