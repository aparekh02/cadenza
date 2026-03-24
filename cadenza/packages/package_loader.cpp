#include "_template/cadenza_package.h"
#include "../ikm/ipc/ikm_bus.h"
#include <dlfcn.h>
#include <cstring>
#include <atomic>

namespace cadenza::packages {

static constexpr int MAX_LOADED = 32;

struct LoadedPackage {
    void* handle;
    CadenzaPackage* instance;
    char name[64];
    char path[256];
    bool active;
};

static LoadedPackage loaded[MAX_LOADED];
static std::atomic<int> loaded_count{0};

int package_load(const char* so_path) {
    if (!so_path) return -1;

    int idx = loaded_count.load(std::memory_order_acquire);
    if (idx >= MAX_LOADED) return -2;

    // dlopen the shared library
    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        // Log error to IKM
        uint32_t event_topic = ikm::ikm_topic_from_name("packages/events");
        const char* err = dlerror();
        if (err) {
            ikm::ikm_publish(event_topic, err, std::strlen(err));
        }
        return -3;
    }

    // Check ABI version before calling on_attach
    auto* abi_fn = reinterpret_cast<const char*(*)()>(dlsym(handle, "cadenza_abi_version"));
    if (!abi_fn) {
        dlclose(handle);
        return -4; // Missing ABI symbol
    }

    const char* pkg_abi = abi_fn();
    if (!pkg_abi || std::strcmp(pkg_abi, CADENZA_VERSION) != 0) {
        // ABI version mismatch — log and reject
        uint32_t event_topic = ikm::ikm_topic_from_name("packages/events");
        char msg[256];
        std::snprintf(msg, sizeof(msg), "ABI mismatch: package=%s, expected=%s",
                      pkg_abi ? pkg_abi : "null", CADENZA_VERSION);
        ikm::ikm_publish(event_topic, msg, std::strlen(msg));
        dlclose(handle);
        return -5; // ABI mismatch
    }

    // Create package instance
    auto* create_fn = reinterpret_cast<CadenzaPackage*(*)()>(
        dlsym(handle, "cadenza_create_package"));
    if (!create_fn) {
        dlclose(handle);
        return -6;
    }

    CadenzaPackage* pkg = create_fn();
    if (!pkg) {
        dlclose(handle);
        return -7;
    }

    // Call on_attach
    int rc = pkg->on_attach();
    if (rc != 0) {
        auto* destroy_fn = reinterpret_cast<void(*)(CadenzaPackage*)>(
            dlsym(handle, "cadenza_destroy_package"));
        if (destroy_fn) destroy_fn(pkg);
        dlclose(handle);
        return -8;
    }

    // Store
    auto& entry = loaded[idx];
    entry.handle = handle;
    entry.instance = pkg;
    std::strncpy(entry.path, so_path, sizeof(entry.path) - 1);

    auto meta = pkg->metadata();
    std::strncpy(entry.name, meta.name, sizeof(entry.name) - 1);
    entry.active = true;

    loaded_count.store(idx + 1, std::memory_order_release);

    // Log successful load
    uint32_t event_topic = ikm::ikm_topic_from_name("packages/events");
    char msg[256];
    std::snprintf(msg, sizeof(msg), "loaded: %s v%s", meta.name, meta.version);
    ikm::ikm_publish(event_topic, msg, std::strlen(msg));

    return 0;
}

int package_unload(const char* name) {
    if (!name) return -1;

    int count = loaded_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (loaded[i].active && std::strcmp(loaded[i].name, name) == 0) {
            // Call on_detach
            if (loaded[i].instance) {
                loaded[i].instance->on_detach();

                auto* destroy_fn = reinterpret_cast<void(*)(CadenzaPackage*)>(
                    dlsym(loaded[i].handle, "cadenza_destroy_package"));
                if (destroy_fn) destroy_fn(loaded[i].instance);
            }

            dlclose(loaded[i].handle);
            loaded[i].active = false;
            loaded[i].instance = nullptr;
            loaded[i].handle = nullptr;

            return 0;
        }
    }

    return -2; // Not found
}

int package_tick_all() {
    int count = loaded_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (loaded[i].active && loaded[i].instance) {
            loaded[i].instance->on_tick();
        }
    }
    return 0;
}

int package_get_count() {
    int active = 0;
    int count = loaded_count.load(std::memory_order_acquire);
    for (int i = 0; i < count; i++) {
        if (loaded[i].active) active++;
    }
    return active;
}

} // namespace cadenza::packages
