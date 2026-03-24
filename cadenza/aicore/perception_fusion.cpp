/**
 * Perception Fusion — merges all sensor streams into a unified WorldState.
 *
 * Subscribes to IKM sensor topics and maintains a continuously-updated
 * world model. Each sensor modality runs at its own rate; fusion runs
 * at the INFERENCE tick rate (50ms) and produces the best estimate
 * from whatever data is freshest.
 *
 * Modalities:
 *   - Proprioception: IMU + joint state → body pose, stability estimate
 *   - Exteroception:  LiDAR/depth → terrain classification, obstacle map
 *   - Vision:         Camera frames → object detection, scene understanding
 *   - Temporal:       History buffer → motion trend, terrain prediction
 */

#include "aicore_api.h"
#include "../ikm/ipc/ikm_bus.h"
#include <atomic>
#include <cstring>
#include <cmath>
#include <chrono>

namespace cadenza::aicore {

static WorldState current_state;
static std::atomic<bool> fusion_active{false};
static std::atomic<uint64_t> last_imu_us{0};
static std::atomic<uint64_t> last_lidar_us{0};

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count());
}

// ── Sensor callbacks ────────────────────────────────────────────────────────

static void on_imu(uint32_t, const void* data, size_t len) {
    if (!data || len < sizeof(float) * 6) return;
    const float* imu = static_cast<const float*>(data);

    auto& body = current_state.body;
    body.linear_accel[0] = imu[0];
    body.linear_accel[1] = imu[1];
    body.linear_accel[2] = imu[2];
    body.angular_vel[0] = imu[3];
    body.angular_vel[1] = imu[4];
    body.angular_vel[2] = imu[5];

    // Complementary filter for orientation
    float dt = 0.01f; // ~100Hz IMU
    body.roll  += imu[3] * dt;
    body.pitch += imu[4] * dt;
    body.yaw   += imu[5] * dt;

    // Gravity-based correction
    float accel_roll  = std::atan2(imu[1], imu[2]);
    float accel_pitch = std::atan2(-imu[0], std::sqrt(imu[1]*imu[1] + imu[2]*imu[2]));
    float alpha = 0.98f;
    body.roll  = alpha * body.roll  + (1.0f - alpha) * accel_roll;
    body.pitch = alpha * body.pitch + (1.0f - alpha) * accel_pitch;

    last_imu_us.store(now_us(), std::memory_order_release);
}

static void on_joint_state(uint32_t, const void* data, size_t len) {
    if (!data || len < sizeof(float) * 24) return;
    const float* joints = static_cast<const float*>(data);
    std::memcpy(current_state.body.joint_q, joints, 12 * sizeof(float));
    std::memcpy(current_state.body.joint_dq, joints + 12, 12 * sizeof(float));
}

static void on_lidar(uint32_t, const void* data, size_t len) {
    if (!data || len < sizeof(float) * 3) return;

    const float* points = static_cast<const float*>(data);
    int n_points = static_cast<int>(len / (3 * sizeof(float)));

    float z_min = 1e6f, z_max = -1e6f;
    float z_sum = 0, z_sq_sum = 0;
    float nearest_obstacle = 100.0f;

    for (int i = 0; i < n_points && i < 1000; i++) {
        float x = points[i * 3];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];

        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
        z_sum += z;
        z_sq_sum += z * z;

        float dist = std::sqrt(x * x + y * y);
        if (z > 0.1f && dist < nearest_obstacle) {
            nearest_obstacle = dist;
            current_state.terrain.obstacle_height = z;
        }
    }

    if (n_points > 0) {
        float z_mean = z_sum / n_points;
        float z_var = z_sq_sum / n_points - z_mean * z_mean;
        current_state.terrain.roughness = std::min(1.0f, std::sqrt(z_var) * 10.0f);
        current_state.terrain.slope = std::atan2(z_max - z_min, 1.0f);
        current_state.terrain.obstacle_distance = nearest_obstacle;

        if (z_var < 0.001f) {
            current_state.terrain.terrain_class = 0; // flat
        } else if (current_state.terrain.slope > 0.15f) {
            current_state.terrain.terrain_class = 3; // slope
        } else if (z_max - z_min > 0.08f) {
            current_state.terrain.terrain_class = 2; // stairs
        } else {
            current_state.terrain.terrain_class = 1; // rough
        }
    }

    last_lidar_us.store(now_us(), std::memory_order_release);
}

static void on_goal(uint32_t, const void* data, size_t len) {
    if (!data || len < sizeof(GoalState)) return;
    std::memcpy(&current_state.goal, data, sizeof(GoalState));
}

// ── Confidence estimation ───────────────────────────────────────────────────

static float compute_confidence() {
    uint64_t now = now_us();
    float conf = 1.0f;

    uint64_t imu_age = now - last_imu_us.load(std::memory_order_acquire);
    if (imu_age > 100000) conf *= 0.5f;
    if (imu_age > 500000) conf *= 0.1f;

    uint64_t lidar_age = now - last_lidar_us.load(std::memory_order_acquire);
    if (lidar_age > 500000) conf *= 0.7f;

    float tilt = std::sqrt(current_state.body.roll * current_state.body.roll +
                           current_state.body.pitch * current_state.body.pitch);
    if (tilt > 0.3f) conf *= 0.6f;
    if (tilt > 0.5f) conf *= 0.2f;

    return conf;
}

// ── Public API ──────────────────────────────────────────────────────────────

int perception_fusion_init() {
    std::memset(&current_state, 0, sizeof(WorldState));
    current_state.terrain.friction = 0.6f;
    current_state.goal.goal_type = 3;
    current_state.confidence = 0.0f;

    uint32_t t_imu = ikm::ikm_topic_from_name("sensors/imu");
    ikm::ikm_subscribe(t_imu, on_imu);

    uint32_t t_joints = ikm::ikm_topic_from_name("sensors/joint_state");
    ikm::ikm_subscribe(t_joints, on_joint_state);

    uint32_t t_lidar = ikm::ikm_topic_from_name("sensors/lidar");
    ikm::ikm_subscribe(t_lidar, on_lidar);

    uint32_t t_goal = ikm::ikm_topic_from_name("aicore/goal");
    ikm::ikm_subscribe(t_goal, on_goal);

    fusion_active.store(true, std::memory_order_release);
    return 0;
}

WorldState perception_fusion_update() {
    current_state.timestamp_us = now_us();
    current_state.confidence = compute_confidence();
    return current_state;
}

void perception_fusion_shutdown() {
    fusion_active.store(false, std::memory_order_release);
}

} // namespace cadenza::aicore
