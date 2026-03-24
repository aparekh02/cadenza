#pragma once
#ifndef CADENZA_KERNEL_H
#define CADENZA_KERNEL_H

/**
 * Cadenza Kernel SDK — Build custom kernels for the Cadenza OS.
 *
 * A kernel is a self-contained module that:
 *   1. Registers as a DSK task (with class, period, deadline, core affinity)
 *   2. Publishes/subscribes to IKM topics for inter-kernel communication
 *   3. Implements on_init(), on_tick(), on_shutdown() lifecycle hooks
 *
 * Example:
 *   class MyPerceptionKernel : public CadenzaKernel {
 *       int on_init() override { ... subscribe to "sensors/camera" ... }
 *       int on_tick() override { ... process frame, publish "perception/objects" ... }
 *       void on_shutdown() override { ... cleanup ... }
 *       KernelConfig config() const override {
 *           return {"my_perception", TaskClass::INFERENCE, 50000, 50000, CoreAffinity::SHARED};
 *       }
 *   };
 *
 * Register with:  CADENZA_REGISTER_KERNEL(MyPerceptionKernel)
 * Build with:     cadenza build (auto-discovers kernels in src/)
 */

#include "../dsk/scheduler/dsk_api.h"
#include "../ikm/ipc/ikm_bus.h"

#include <cstdint>
#include <string>

namespace cadenza::sdk {

struct KernelConfig {
    const char* name;
    dsk::TaskClass task_class;
    uint32_t period_us;
    uint32_t deadline_us;
    dsk::CoreAffinity core;
    int priority = 40; // default: INFERENCE priority
};

class CadenzaKernel {
public:
    virtual ~CadenzaKernel() = default;

    /// Called once at boot. Subscribe to IKM topics, allocate resources.
    virtual int on_init() = 0;

    /// Called every period. Do your work here.
    virtual int on_tick() = 0;

    /// Called at shutdown. Release resources.
    virtual void on_shutdown() = 0;

    /// Return this kernel's scheduling configuration.
    virtual KernelConfig config() const = 0;

    // ── Convenience helpers ──

    /// Publish data to an IKM topic.
    int publish(const char* topic, const void* data, size_t len) {
        uint32_t tid = ikm::ikm_topic_from_name(topic);
        return ikm::ikm_publish(tid, data, len);
    }

    /// Subscribe to an IKM topic.
    int subscribe(const char* topic, ikm::SubscriberCallback callback) {
        uint32_t tid = ikm::ikm_topic_from_name(topic);
        return ikm::ikm_subscribe(tid, callback);
    }

    /// Register this kernel with DSK. Called by the framework.
    int register_with_dsk() {
        auto cfg = config();
        dsk::CadenzaTask task{};
        task.task_id = next_task_id_++;
        task.task_class = cfg.task_class;
        task.core = cfg.core;
        task.period_us = cfg.period_us;
        task.deadline_us = cfg.deadline_us;
        task.priority = cfg.priority;
        task.entry = [this]() { this->on_tick(); };
        task_id_ = task.task_id;
        return dsk::dsk_register_task(&task);
    }

    uint32_t task_id() const { return task_id_; }

private:
    uint32_t task_id_ = 0;
    static inline uint32_t next_task_id_ = 2000;
};

} // namespace cadenza::sdk

/// Register a kernel class for auto-discovery by `cadenza build`.
#define CADENZA_REGISTER_KERNEL(KernelClass) \
    extern "C" cadenza::sdk::CadenzaKernel* cadenza_create_kernel() { \
        return new KernelClass(); \
    } \
    extern "C" void cadenza_destroy_kernel(cadenza::sdk::CadenzaKernel* k) { \
        delete k; \
    }

#endif
