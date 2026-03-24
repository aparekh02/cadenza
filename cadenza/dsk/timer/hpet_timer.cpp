#include <cstdint>
#include <cerrno>
#include <thread>
#include <chrono>

#ifdef __linux__
#include <time.h>
#endif

namespace cadenza::dsk {

// Microsecond-precision periodic wakeup using clock_nanosleep(CLOCK_MONOTONIC)
int hpet_sleep_us(uint64_t microseconds) {
#ifdef __linux__
    struct timespec ts{};
    ts.tv_sec = static_cast<time_t>(microseconds / 1000000ULL);
    ts.tv_nsec = static_cast<long>((microseconds % 1000000ULL) * 1000L);

    int rc;
    do {
        rc = clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts);
    } while (rc == EINTR);

    return rc;
#else
    // Fallback for non-Linux (test mode)
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return 0;
#endif
}

// Absolute time sleep — wakes at a specific point in time
int hpet_sleep_until(uint64_t target_ns) {
#ifdef __linux__
    struct timespec ts{};
    ts.tv_sec = static_cast<time_t>(target_ns / 1000000000ULL);
    ts.tv_nsec = static_cast<long>(target_ns % 1000000000ULL);

    int rc;
    do {
        rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
    } while (rc == EINTR);

    return rc;
#else
    auto now = std::chrono::steady_clock::now();
    auto now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
    if (target_ns > now_ns) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(target_ns - now_ns));
    }
    return 0;
#endif
}

// Get current monotonic time in nanoseconds
uint64_t hpet_now_ns() {
#ifdef __linux__
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
#else
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
#endif
}

} // namespace cadenza::dsk
