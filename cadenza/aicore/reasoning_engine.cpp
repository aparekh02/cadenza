/**
 * Reasoning Engine — the decision-making core.
 *
 * Takes a fused WorldState and produces an ActionDecision.
 *
 * Architecture: Behavior tree with reactive priority layers.
 *
 *   Layer 0 (SAFETY):   Immediate reflexes — tilt recovery, obstacle halt
 *   Layer 1 (REACTIVE):  Terrain adaptation — gait selection, speed adjustment
 *   Layer 2 (TACTICAL):  Goal pursuit — path following, waypoint navigation
 *   Layer 3 (STRATEGIC): Goal planning — when no immediate plan, what to do
 *
 * Each layer can override lower-priority layers. Safety always wins.
 *
 * The engine is deterministic and inspectable — every decision includes
 * a reasoning string explaining why. No black-box neural network here;
 * neural perception feeds INTO this engine, not replaces it.
 *
 * Optional: delegates to a local SLM (via IPC to the Python aicore modules)
 * for complex reasoning that pure rules can't handle.
 */

#include "aicore_api.h"
#include "../ikm/ipc/ikm_bus.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <atomic>

namespace cadenza::aicore {

static ActionDecision last_decision;
static std::atomic<uint32_t> decision_count{0};
static uint32_t decision_topic_id = 0;

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

// ── Layer 0: Safety reflexes ────────────────────────────────────────────────

static bool check_safety(const WorldState& ws, ActionDecision& out) {
    float tilt = std::sqrt(ws.body.roll * ws.body.roll +
                           ws.body.pitch * ws.body.pitch);

    if (tilt > 0.45f) {
        std::strncpy(out.action_name, "stand", sizeof(out.action_name));
        out.speed = 0.5f;
        out.priority = 3;
        out.confidence = 0.95f;
        std::snprintf(out.reasoning, sizeof(out.reasoning),
                      "SAFETY: critical tilt %.1f deg, stabilizing",
                      tilt * 57.2958f);
        return true;
    }

    if (ws.terrain.obstacle_distance < 0.3f &&
        ws.terrain.obstacle_distance > 0.0f) {
        std::strncpy(out.action_name, "stand", sizeof(out.action_name));
        out.speed = 1.0f;
        out.priority = 3;
        out.confidence = 0.9f;
        std::snprintf(out.reasoning, sizeof(out.reasoning),
                      "SAFETY: obstacle at %.2fm, halting",
                      ws.terrain.obstacle_distance);
        return true;
    }

    int contacts = 0;
    for (int i = 0; i < 4; i++) contacts += ws.body.foot_contact[i];
    if (contacts <= 1 && ws.body.height > 0.1f) {
        std::strncpy(out.action_name, "stand", sizeof(out.action_name));
        out.speed = 0.3f;
        out.priority = 3;
        out.confidence = 0.85f;
        std::snprintf(out.reasoning, sizeof(out.reasoning),
                      "SAFETY: only %d feet in contact, stabilizing", contacts);
        return true;
    }

    return false;
}

// ── Layer 1: Terrain-reactive gait selection ────────────────────────────────

static bool check_terrain(const WorldState& ws, ActionDecision& out) {
    if (ws.confidence < 0.3f) return false;

    float base_speed = 1.0f;
    const char* gait = "walk_forward";
    const char* reason_fmt = nullptr;

    switch (ws.terrain.terrain_class) {
        case 0:
            gait = "walk_forward";
            base_speed = 1.0f;
            break;
        case 1:
            gait = "crawl_forward";
            base_speed = 0.6f;
            reason_fmt = "TERRAIN: rough surface (roughness=%.2f), crawling";
            break;
        case 2:
            gait = "climb_step";
            base_speed = 0.4f;
            reason_fmt = "TERRAIN: stairs detected (height=%.2fm), climbing";
            break;
        case 3:
            if (ws.terrain.slope > 0) {
                gait = "crawl_forward";
                base_speed = 0.5f;
            } else {
                gait = "walk_forward";
                base_speed = 0.7f;
            }
            reason_fmt = "TERRAIN: slope %.1f deg, adapting gait";
            break;
        case 4:
            gait = "jump";
            base_speed = 1.5f;
            reason_fmt = "TERRAIN: gap detected, jumping";
            break;
        default:
            return false;
    }

    if (ws.goal.goal_type == 3) return false;

    std::strncpy(out.action_name, gait, sizeof(out.action_name));
    out.speed = base_speed;
    out.priority = 1;
    out.confidence = ws.confidence * 0.8f;

    if (reason_fmt) {
        float val = (ws.terrain.terrain_class == 1) ? ws.terrain.roughness :
                    (ws.terrain.terrain_class == 2) ? ws.terrain.obstacle_height :
                    ws.terrain.slope * 57.2958f;
        std::snprintf(out.reasoning, sizeof(out.reasoning), reason_fmt, val);
    } else {
        std::strncpy(out.reasoning, "TERRAIN: flat ground, normal gait",
                      sizeof(out.reasoning));
    }

    return true;
}

// ── Layer 2: Goal pursuit ───────────────────────────────────────────────────

static bool check_goal(const WorldState& ws, ActionDecision& out) {
    if (ws.goal.goal_type == 3) return false;

    float tx = ws.goal.target_position[0];
    float ty = ws.goal.target_position[1];
    if (tx == 0 && ty == 0) {
        return false;
    }

    float dist = std::sqrt(tx * tx + ty * ty);
    float heading = std::atan2(ty, tx);

    if (dist < 0.2f) {
        std::strncpy(out.action_name, "stand", sizeof(out.action_name));
        out.speed = 1.0f;
        out.priority = 1;
        out.confidence = 0.9f;
        std::snprintf(out.reasoning, sizeof(out.reasoning),
                      "GOAL: arrived at target (%.2fm away)", dist);
        return true;
    }

    if (std::fabs(heading) > 0.3f) {
        std::strncpy(out.action_name,
                      heading > 0 ? "turn_left" : "turn_right",
                      sizeof(out.action_name));
        out.speed = 0.6f;
        out.heading_rad = heading;
        out.priority = 1;
        out.confidence = 0.8f;
        std::snprintf(out.reasoning, sizeof(out.reasoning),
                      "GOAL: turning %.0f deg toward target (%.1fm away)",
                      heading * 57.2958f, dist);
        return true;
    }

    std::strncpy(out.action_name, "walk_forward", sizeof(out.action_name));
    out.speed = std::min(1.0f, ws.goal.urgency + 0.5f);
    out.distance_m = dist;
    out.priority = 1;
    out.confidence = 0.8f;
    std::snprintf(out.reasoning, sizeof(out.reasoning),
                  "GOAL: walking toward target (%.1fm away)", dist);
    return true;
}

// ── Layer 3: Strategic / idle behavior ──────────────────────────────────────

static void default_behavior(const WorldState& ws, ActionDecision& out) {
    if (ws.goal.goal_type == 2) {
        std::strncpy(out.action_name, "walk_forward", sizeof(out.action_name));
        out.speed = 0.5f;
        out.priority = 0;
        out.confidence = 0.5f;
        std::strncpy(out.reasoning, "STRATEGIC: exploring",
                      sizeof(out.reasoning));
    } else {
        std::strncpy(out.action_name, "stand", sizeof(out.action_name));
        out.speed = 1.0f;
        out.priority = 0;
        out.confidence = 1.0f;
        std::strncpy(out.reasoning, "IDLE: no goal, standing by",
                      sizeof(out.reasoning));
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

int reasoning_engine_init() {
    std::memset(&last_decision, 0, sizeof(ActionDecision));
    decision_count.store(0, std::memory_order_release);
    decision_topic_id = ikm::ikm_topic_from_name("aicore/decision");
    return 0;
}

ActionDecision reasoning_engine_decide(const WorldState& ws) {
    ActionDecision decision{};
    decision.timestamp_us = now_us();

    if (!check_safety(ws, decision)) {
        if (!check_terrain(ws, decision)) {
            if (!check_goal(ws, decision)) {
                default_behavior(ws, decision);
            }
        }
    }

    ikm::ikm_publish(decision_topic_id, &decision, sizeof(decision));

    last_decision = decision;
    decision_count.fetch_add(1, std::memory_order_relaxed);
    return decision;
}

ActionDecision reasoning_engine_get_last() {
    return last_decision;
}

uint32_t reasoning_engine_get_count() {
    return decision_count.load(std::memory_order_acquire);
}

} // namespace cadenza::aicore
