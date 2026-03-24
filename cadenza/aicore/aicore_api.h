#pragma once
#ifndef CADENZA_AICORE_API_H
#define CADENZA_AICORE_API_H

/**
 * AICore — Lightweight multimodal intelligence kernel.
 *
 * The decision-making center of the Cadenza OS. Fuses sensor streams
 * (vision, proprioception, terrain, language goals) into a unified world
 * state, reasons about what the robot should do next, and publishes
 * action decisions that the control layer executes.
 *
 * Runs as a DSK INFERENCE task (50ms period, SHARED cores).
 *
 * IKM topics consumed:
 *   sensors/imu           — body orientation, acceleration
 *   sensors/lidar         — point cloud / depth
 *   sensors/camera        — vision frames
 *   sensors/joint_state   — current joint angles + velocities
 *   aicore/goal           — high-level goal from developer or planner
 *
 * IKM topics produced:
 *   aicore/world_state    — fused world state (structured)
 *   aicore/decision       — action decision (what to do next)
 *   aicore/status         — kernel health, model latency, confidence
 */

#include <cstdint>
#include <cstddef>

namespace cadenza::aicore {

// ── World State ─────────────────────────────────────────────────────────────

struct BodyState {
    float roll, pitch, yaw;          // radians
    float angular_vel[3];            // rad/s
    float linear_accel[3];           // m/s^2
    float height;                    // meters above ground
    float joint_q[12];              // joint positions
    float joint_dq[12];             // joint velocities
    uint8_t foot_contact[4];         // 0/1 per foot
};

struct TerrainState {
    float slope;                     // radians
    float roughness;                 // 0-1
    float friction;                  // estimated coefficient
    float obstacle_distance;         // nearest obstacle (meters)
    float obstacle_height;           // meters
    uint8_t terrain_class;           // 0=flat 1=rough 2=stairs 3=slope 4=gap
};

struct GoalState {
    char goal_text[256];             // "walk to the door", "explore", "follow me"
    float target_position[3];        // optional waypoint (0,0,0 = none)
    float urgency;                   // 0-1 (0=casual, 1=emergency)
    uint8_t goal_type;               // 0=navigate 1=interact 2=explore 3=idle
};

struct WorldState {
    uint64_t timestamp_us;
    BodyState body;
    TerrainState terrain;
    GoalState goal;
    float confidence;                // 0-1 overall state confidence
};

// ── Decision ────────────────────────────────────────────────────────────────

struct ActionDecision {
    uint64_t timestamp_us;
    char action_name[64];            // "walk_forward", "turn_left", etc.
    float speed;                     // speed multiplier
    float heading_rad;               // desired heading change
    float distance_m;                // desired distance (0 = continuous)
    float confidence;                // 0-1 decision confidence
    uint8_t priority;                // 0=low 1=normal 2=high 3=critical
    char reasoning[256];             // human-readable explanation
};

struct AICoreStatus {
    uint64_t timestamp_us;
    uint32_t decisions_made;
    uint32_t perception_latency_us;
    uint32_t reasoning_latency_us;
    float avg_confidence;
    bool model_loaded;
    char model_name[64];
};

// ── API ─────────────────────────────────────────────────────────────────────

int aicore_init(const char* model_path = nullptr);
int aicore_start();
void aicore_stop();
void aicore_shutdown();

int aicore_set_goal(const char* goal_text, const float* target_pos = nullptr,
                    float urgency = 0.5f);

WorldState aicore_get_world_state();
ActionDecision aicore_get_last_decision();
AICoreStatus aicore_get_status();

} // namespace cadenza::aicore

#endif
