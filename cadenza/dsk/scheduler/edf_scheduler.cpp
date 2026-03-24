#include "dsk_api.h"
#include <cstring>
#include <pthread.h>
#include <atomic>

#ifdef __linux__
#include <linux/sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace cadenza::dsk {

// sched_setattr syscall wrapper for SCHED_DEADLINE
#ifdef __linux__
struct sched_attr_t {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
};

static int sched_setattr_wrapper(pid_t pid, const sched_attr_t* attr, unsigned int flags) {
    return static_cast<int>(syscall(SYS_sched_setattr, pid, attr, flags));
}
#endif

int edf_schedule_task(const CadenzaTask* task, pthread_t thread) {
    if (!task) return -1;

    // Only INFERENCE and SENSOR use SCHED_DEADLINE (EDF)
    if (task->task_class != TaskClass::INFERENCE && task->task_class != TaskClass::SENSOR) {
        return -2;
    }

#if defined(__linux__) && !defined(CADENZA_TEST_MODE)
    sched_attr_t attr{};
    attr.size = sizeof(sched_attr_t);
    attr.sched_policy = 6; // SCHED_DEADLINE
    attr.sched_flags = 0;

    // Convert microseconds to nanoseconds
    attr.sched_runtime  = static_cast<uint64_t>(task->period_us) * 500ULL; // 50% of period
    attr.sched_deadline = static_cast<uint64_t>(task->deadline_us) * 1000ULL;
    attr.sched_period   = static_cast<uint64_t>(task->period_us) * 1000ULL;

    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    int rc = sched_setattr_wrapper(tid, &attr, 0);
    if (rc != 0) return -3;
#else
    (void)thread;
#endif

    return 0;
}

} // namespace cadenza::dsk
