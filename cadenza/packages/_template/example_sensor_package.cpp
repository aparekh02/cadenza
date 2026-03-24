#include "cadenza_package.h"
#include "../../ikm/ipc/ikm_bus.h"
#include <cstring>
#include <cmath>

class ExampleSensorPackage : public cadenza::packages::CadenzaPackage {
public:
    int on_attach() override {
        topic_id_ = cadenza::ikm::ikm_topic_from_name("sensors/imu_external");
        tick_count_ = 0;
        return 0;
    }

    int on_tick() override {
        // Publish synthetic IMU data
        float imu_data[6] = {
            0.01f * std::sin(static_cast<float>(tick_count_) * 0.1f),  // ax
            0.02f * std::cos(static_cast<float>(tick_count_) * 0.1f),  // ay
            9.81f,   // az
            0.001f,  // gx
            0.002f,  // gy
            0.0f     // gz
        };

        cadenza::ikm::ikm_publish(topic_id_, imu_data, sizeof(imu_data));
        tick_count_++;
        return 0;
    }

    int on_detach() override {
        return 0;
    }

    cadenza::packages::PackageMetadata metadata() const override {
        static const char* pub_topics[] = {"sensors/imu_external"};
        return {
            .name = "example_sensor",
            .version = "0.1.0",
            .tick_rate_ms = 10,
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
        return new ExampleSensorPackage();
    }
    void cadenza_destroy_package(cadenza::packages::CadenzaPackage* pkg) {
        delete pkg;
    }
}
