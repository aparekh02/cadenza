#include "dsk_api.h"
#include <pthread.h>

namespace cadenza::dsk {

int rms_schedule_task(const CadenzaTask* task, pthread_t thread) {
    if (!task) return -1;

    // Only SAFETY and CONTROL use SCHED_FIFO (RMS)
    if (task->task_class != TaskClass::SAFETY && task->task_class != TaskClass::CONTROL) {
        return -2;
    }

#if defined(__linux__) && !defined(CADENZA_TEST_MODE)
    struct sched_param param{};
    param.sched_priority = task->priority;

    int rc = pthread_setschedparam(thread, SCHED_FIFO, &param);
    if (rc != 0) return -3;
#else
    (void)thread;
#endif

    return 0;
}

// Set core affinity for the task thread
int set_core_affinity(const CadenzaTask* task, pthread_t thread) {
    if (!task) return -1;

#if defined(__linux__) && !defined(CADENZA_TEST_MODE)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    switch (task->core) {
        case CoreAffinity::ISOLATED_1:
            CPU_SET(2, &cpuset); // Core 2 (isolated via kernel boot param)
            break;
        case CoreAffinity::ISOLATED_2:
            CPU_SET(3, &cpuset); // Core 3 (isolated via kernel boot param)
            break;
        case CoreAffinity::SHARED:
            CPU_SET(0, &cpuset);
            CPU_SET(1, &cpuset);
            break;
    }

    return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#else
    (void)thread;
    return 0;
#endif
}

} // namespace cadenza::dsk
