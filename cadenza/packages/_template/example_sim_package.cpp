#include "cadenza_package.h"
#include "../../ikm/ipc/ikm_bus.h"
#include <cstring>
#include <cmath>

// This package publishes to IKM topic "sensors/lidar" — the IDENTICAL topic
// that a real LiDAR sensor package uses. RLAK subscribes to this topic and
// cannot distinguish sim data from real data — this is BY DESIGN.
// This allows training on synthetic data without any modification to RLAK.

class ExampleSimPackage : public cadenza::packages::CadenzaPackage {
public:
    int on_attach() override {
        topic_id_ = cadenza::ikm::ikm_topic_from_name("sensors/lidar");
        tick_count_ = 0;
        return 0;
    }

    int on_tick() override {
        // Generate synthetic LiDAR point cloud
        // 100 points in a semicircle pattern
        static constexpr int NUM_POINTS = 100;
        float points[NUM_POINTS * 3]; // x, y, z

        for (int i = 0; i < NUM_POINTS; i++) {
            float angle = static_cast<float>(i) / static_cast<float>(NUM_POINTS) *
                          static_cast<float>(M_PI);
            float range = 2.0f + 0.5f * std::sin(static_cast<float>(tick_count_) * 0.05f);
            points[i * 3]     = range * std::cos(angle); // x
            points[i * 3 + 1] = range * std::sin(angle); // y
            points[i * 3 + 2] = 0.1f;                     // z (ground level)
        }

        cadenza::ikm::ikm_publish(topic_id_, points, sizeof(points));
        tick_count_++;
        return 0;
    }

    int on_detach() override {
        return 0;
    }

    cadenza::packages::PackageMetadata metadata() const override {
        static const char* pub_topics[] = {"sensors/lidar"};
        return {
            .name = "example_sim",
            .version = "0.1.0",
            .tick_rate_ms = 50,
            .preferred_core = 0,
            .topics_published = pub_topics,
            .num_topics_published = 1,
            .topics_subscribed = nullptr,
            .num_topics_subscribed = 0,
        };
    }

private:
    uint32_t topic_id_ = 0;
    uint64_t tick_count_ = 0;
};

extern "C" {
    const char* cadenza_abi_version() { return CADENZA_VERSION; }
    cadenza::packages::CadenzaPackage* cadenza_create_package() {
        return new ExampleSimPackage();
    }
    void cadenza_destroy_package(cadenza::packages::CadenzaPackage* pkg) {
        delete pkg;
    }
}
