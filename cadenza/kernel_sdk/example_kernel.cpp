/**
 * Example custom kernel — demonstrates the Cadenza Kernel SDK.
 *
 * This kernel subscribes to "sensors/imu_external" and publishes
 * a filtered orientation estimate to "perception/orientation".
 *
 * Build:  cadenza build
 * Attach: cadenza attach my_orientation_kernel
 */

#include "cadenza_kernel.h"
#include <cmath>
#include <cstring>
#include <atomic>

namespace {

struct OrientationEstimate {
    float roll;
    float pitch;
    float yaw;
    uint64_t timestamp_us;
};

class OrientationKernel : public cadenza::sdk::CadenzaKernel {
public:
    int on_init() override {
        // Subscribe to IMU data from a sensor package
        subscribe("sensors/imu_external",
            [this](uint32_t, const void* data, size_t len) {
                if (data && len >= 6 * sizeof(float)) {
                    const float* imu = static_cast<const float*>(data);
                    // Simple complementary filter
                    float alpha = 0.98f;
                    roll_  = alpha * (roll_  + imu[3] * 0.01f) + (1 - alpha) * std::atan2(imu[1], imu[2]);
                    pitch_ = alpha * (pitch_ + imu[4] * 0.01f) + (1 - alpha) * std::atan2(-imu[0], imu[2]);
                    updated_.store(true, std::memory_order_release);
                }
            });
        return 0;
    }

    int on_tick() override {
        if (!updated_.load(std::memory_order_acquire)) return 0;

        OrientationEstimate est{};
        est.roll = roll_;
        est.pitch = pitch_;
        est.yaw = yaw_;

        publish("perception/orientation", &est, sizeof(est));
        updated_.store(false, std::memory_order_release);
        return 0;
    }

    void on_shutdown() override {
        // Nothing to clean up
    }

    cadenza::sdk::KernelConfig config() const override {
        return {
            .name = "orientation_filter",
            .task_class = cadenza::dsk::TaskClass::SENSOR,
            .period_us = 10000,     // 10ms (100 Hz)
            .deadline_us = 10000,
            .core = cadenza::dsk::CoreAffinity::ISOLATED_1,
            .priority = 60,
        };
    }

private:
    float roll_ = 0, pitch_ = 0, yaw_ = 0;
    std::atomic<bool> updated_{false};
};

} // anonymous namespace

CADENZA_REGISTER_KERNEL(OrientationKernel)
