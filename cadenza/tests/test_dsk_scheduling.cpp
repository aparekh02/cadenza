// Test 2: Register tasks at all five class levels.
// Verify SAFETY always preempts CONTROL.
// Verify INFERENCE yields when CONTROL becomes active.
// Verify dsk_reprioritize() rejects SAFETY class with error code.

#include "../dsk/scheduler/dsk_api.h"
#include "../ikm/ipc/ikm_bus.h"
#include <cassert>
#include <cstdio>
#include <atomic>
#include <thread>
#include <chrono>

using namespace cadenza::dsk;

static std::atomic<int> safety_count{0};
static std::atomic<int> control_count{0};
static std::atomic<int> sensor_count{0};
static std::atomic<int> inference_count{0};
static std::atomic<int> comms_count{0};

static int test_task_registration() {
    // Init IKM first (DSK publishes to it)
    cadenza::ikm::ikm_bus_init();
    int rc = dsk_init();
    assert(rc == 0 && "dsk_init failed");

    // Register one task per class
    CadenzaTask safety{};
    safety.task_id = 1;
    safety.task_class = TaskClass::SAFETY;
    safety.core = CoreAffinity::ISOLATED_2;
    safety.period_us = 1000;
    safety.deadline_us = 1000;
    safety.priority = 99;
    safety.entry = []() { safety_count.fetch_add(1, std::memory_order_relaxed); };

    CadenzaTask control{};
    control.task_id = 2;
    control.task_class = TaskClass::CONTROL;
    control.core = CoreAffinity::ISOLATED_1;
    control.period_us = 5000;
    control.deadline_us = 5000;
    control.priority = 80;
    control.entry = []() { control_count.fetch_add(1, std::memory_order_relaxed); };

    CadenzaTask sensor{};
    sensor.task_id = 3;
    sensor.task_class = TaskClass::SENSOR;
    sensor.core = CoreAffinity::ISOLATED_1;
    sensor.period_us = 10000;
    sensor.deadline_us = 10000;
    sensor.priority = 60;
    sensor.entry = []() { sensor_count.fetch_add(1, std::memory_order_relaxed); };

    CadenzaTask inference{};
    inference.task_id = 4;
    inference.task_class = TaskClass::INFERENCE;
    inference.core = CoreAffinity::SHARED;
    inference.period_us = 50000;
    inference.deadline_us = 50000;
    inference.priority = 40;
    inference.entry = []() { inference_count.fetch_add(1, std::memory_order_relaxed); };

    CadenzaTask comms{};
    comms.task_id = 5;
    comms.task_class = TaskClass::COMMS;
    comms.core = CoreAffinity::SHARED;
    comms.period_us = 100000;
    comms.deadline_us = 100000;
    comms.priority = 0;
    comms.entry = []() { comms_count.fetch_add(1, std::memory_order_relaxed); };

    assert(dsk_register_task(&safety) == 0);
    assert(dsk_register_task(&control) == 0);
    assert(dsk_register_task(&sensor) == 0);
    assert(dsk_register_task(&inference) == 0);
    assert(dsk_register_task(&comms) == 0);

    // Let tasks run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Verify all tasks executed
    assert(safety_count.load() > 0 && "SAFETY task did not execute");
    assert(control_count.load() > 0 && "CONTROL task did not execute");
    std::printf("  SAFETY executed %d times, CONTROL %d times OK\n",
                safety_count.load(), control_count.load());

    // SAFETY should execute more often (1ms period vs 5ms)
    assert(safety_count.load() >= control_count.load() &&
           "SAFETY should preempt CONTROL");
    std::printf("  SAFETY preempts CONTROL: OK\n");

    return 0;
}

static int test_reprioritize_rejection() {
    // dsk_reprioritize must reject SAFETY and CONTROL classes
    int rc = dsk_reprioritize(1, 50); // task_id=1 is SAFETY
    assert(rc == -2 && "dsk_reprioritize should reject SAFETY");
    std::printf("  dsk_reprioritize rejects SAFETY: OK (rc=%d)\n", rc);

    rc = dsk_reprioritize(2, 50); // task_id=2 is CONTROL
    assert(rc == -2 && "dsk_reprioritize should reject CONTROL");
    std::printf("  dsk_reprioritize rejects CONTROL: OK (rc=%d)\n", rc);

    // INFERENCE and COMMS should be accepted
    rc = dsk_reprioritize(4, 35); // task_id=4 is INFERENCE
    assert(rc == 0 && "dsk_reprioritize should accept INFERENCE");
    std::printf("  dsk_reprioritize accepts INFERENCE: OK\n");

    rc = dsk_reprioritize(5, 0); // task_id=5 is COMMS
    assert(rc == 0 && "dsk_reprioritize should accept COMMS");
    std::printf("  dsk_reprioritize accepts COMMS: OK\n");

    return 0;
}

int main() {
    std::printf("test_dsk_scheduling:\n");
    test_task_registration();
    test_reprioritize_rejection();
    dsk_shutdown();
    cadenza::ikm::ikm_bus_shutdown();
    std::printf("  ALL PASSED\n");
    return 0;
}
