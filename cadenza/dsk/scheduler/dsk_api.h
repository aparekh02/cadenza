#pragma once
#ifndef CADENZA_DSK_API_H
#define CADENZA_DSK_API_H

#include <cstdint>
#include <functional>

namespace cadenza::dsk {

enum class TaskClass : uint8_t {
    SAFETY = 0,
    CONTROL = 1,
    SENSOR = 2,
    INFERENCE = 3,
    COMMS = 4
};

enum class CoreAffinity : uint8_t {
    ISOLATED_1 = 0,
    ISOLATED_2 = 1,
    SHARED = 2
};

struct CadenzaTask {
    uint32_t task_id;
    TaskClass task_class;
    CoreAffinity core;
    uint32_t period_us;
    uint32_t deadline_us;
    int priority;
    std::function<void()> entry;
    bool active;
};

int dsk_init();
int dsk_register_task(CadenzaTask* task);
int dsk_cancel_task(uint32_t task_id);
int dsk_reprioritize(uint32_t task_id, int new_priority);
bool dsk_has_active_control_tasks();
void dsk_shutdown();

} // namespace cadenza::dsk

#endif
