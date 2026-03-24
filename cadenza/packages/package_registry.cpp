#include "_template/cadenza_package.h"
#include "../ikm/ipc/ikm_bus.h"
#include <cstring>
#include <atomic>

namespace cadenza::packages {

static constexpr int MAX_REGISTERED = 64;

struct RegistryEntry {
    char name[64];
    char version[32];
    uint32_t tick_rate_ms;
    uint8_t preferred_core;
    bool healthy;
    bool active;
    uint64_t tick_count;
    uint64_t error_count;
};

static RegistryEntry registry[MAX_REGISTERED];
static std::atomic<int> registry_count{0};

int package_registry_init() {
    std::memset(registry, 0, sizeof(registry));
    registry_count.store(0, std::memory_order_release);
    return 0;
}

int package_registry_add(const PackageMetadata& meta) {
    int idx = registry_count.load(std::memory_order_acquire);
    if (idx >= MAX_REGISTERED) return -1;

    auto& entry = registry[idx];
    std::strncpy(entry.name, meta.name, sizeof(entry.name) - 1);
    std::strncpy(entry.version, meta.version, sizeof(entry.version) - 1);
    entry.tick_rate_ms = meta.tick_rate_ms;
    entry.preferred_core = meta.preferred_core;
    entry.healthy = true;
    entry.active = true;
    entry.tick_count = 0;
    entry.error_count = 0;

    registry_count.store(idx + 1, std::memory_order_release);
    return 0;
}

int package_registry_remove(const char* name) {
    int count = registry_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (registry[i].active && std::strcmp(registry[i].name, name) == 0) {
            registry[i].active = false;
            registry[i].healthy = false;
            return 0;
        }
    }
    return -1;
}

void package_registry_report_health() {
    uint32_t topic = ikm::ikm_topic_from_name("packages/health");
    int count = registry_count.load(std::memory_order_acquire);

    struct HealthReport {
        int total_packages;
        int healthy_packages;
        int active_packages;
    };

    HealthReport report{};
    for (int i = 0; i < count; i++) {
        report.total_packages++;
        if (registry[i].active) report.active_packages++;
        if (registry[i].healthy) report.healthy_packages++;
    }

    ikm::ikm_publish(topic, &report, sizeof(report));
}

} // namespace cadenza::packages
