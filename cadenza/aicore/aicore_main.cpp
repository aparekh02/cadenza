/**
 * AICore Main — kernel lifecycle and DSK integration.
 *
 * Boots the perception fusion and reasoning engine, registers as a
 * DSK INFERENCE task, and runs the sense→think→act loop every 50ms.
 */

#include "aicore_api.h"
#include "../ikm/ipc/ikm_bus.h"
#include "../dsk/scheduler/dsk_api.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>

namespace cadenza::aicore {

// Forward declarations
int perception_fusion_init();
WorldState perception_fusion_update();
void perception_fusion_shutdown();
int reasoning_engine_init();
ActionDecision reasoning_engine_decide(const WorldState& ws);
ActionDecision reasoning_engine_get_last();
uint32_t reasoning_engine_get_count();

static std::atomic<bool> aicore_running{false};
static std::thread core_thread;
static AICoreStatus status;
static uint32_t status_topic_id = 0;
static uint32_t world_state_topic_id = 0;

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

// ── Core loop ───────────────────────────────────────────────────────────────

static void aicore_loop() {
    while (aicore_running.load(std::memory_order_acquire)) {
        uint64_t tick_start = now_us();

        // 1. SENSE
        uint64_t perc_start = now_us();
        WorldState ws = perception_fusion_update();
        uint64_t perc_end = now_us();

        ikm::ikm_publish(world_state_topic_id, &ws, sizeof(ws));

        // 2. THINK
        uint64_t reason_start = now_us();
        ActionDecision decision = reasoning_engine_decide(ws);
        uint64_t reason_end = now_us();

        (void)decision;

        // 3. Update status
        status.timestamp_us = now_us();
        status.decisions_made = reasoning_engine_get_count();
        status.perception_latency_us = static_cast<uint32_t>(perc_end - perc_start);
        status.reasoning_latency_us = static_cast<uint32_t>(reason_end - reason_start);
        status.avg_confidence = ws.confidence;

        ikm::ikm_publish(status_topic_id, &status, sizeof(status));

        // Sleep for remainder of 50ms period
        uint64_t elapsed = now_us() - tick_start;
        if (elapsed < 50000) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(50000 - elapsed));
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

int aicore_init(const char* model_path) {
    std::memset(&status, 0, sizeof(AICoreStatus));

    status_topic_id = ikm::ikm_topic_from_name("aicore/status");
    world_state_topic_id = ikm::ikm_topic_from_name("aicore/world_state");

    perception_fusion_init();
    reasoning_engine_init();

    if (model_path) {
        std::strncpy(status.model_name, model_path,
                     sizeof(status.model_name) - 1);
        status.model_loaded = true;
    } else {
        std::strncpy(status.model_name, "builtin_behavior_tree",
                     sizeof(status.model_name) - 1);
        status.model_loaded = true;
    }

    return 0;
}

int aicore_start() {
    if (aicore_running.load(std::memory_order_acquire)) return -1;
    aicore_running.store(true, std::memory_order_release);
    core_thread = std::thread(aicore_loop);
    return 0;
}

void aicore_stop() {
    aicore_running.store(false, std::memory_order_release);
    if (core_thread.joinable()) {
        core_thread.join();
    }
}

void aicore_shutdown() {
    aicore_stop();
    perception_fusion_shutdown();
}

int aicore_set_goal(const char* goal_text, const float* target_pos,
                    float urgency) {
    GoalState goal{};
    if (goal_text) {
        std::strncpy(goal.goal_text, goal_text, sizeof(goal.goal_text) - 1);
    }
    if (target_pos) {
        goal.target_position[0] = target_pos[0];
        goal.target_position[1] = target_pos[1];
        goal.target_position[2] = target_pos[2];
        goal.goal_type = 0;
    } else if (goal_text && std::strlen(goal_text) > 0) {
        goal.goal_type = 0;
    } else {
        goal.goal_type = 3;
    }
    goal.urgency = urgency;

    uint32_t goal_topic = ikm::ikm_topic_from_name("aicore/goal");
    return ikm::ikm_publish(goal_topic, &goal, sizeof(goal));
}

WorldState aicore_get_world_state() {
    return perception_fusion_update();
}

ActionDecision aicore_get_last_decision() {
    return reasoning_engine_get_last();
}

AICoreStatus aicore_get_status() {
    return status;
}

} // namespace cadenza::aicore
