// cadenza kernel
// Displays a live table of DSK kernel tasks:
//   name | core | CPU% | miss rate | latency

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);
extern std::string ssh_prefix(const std::string& host);
extern int connect_local();

// Pretty-print the kernel task table header
static void print_header() {
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "TASK", "CLASS", "CORE", "CPU%", "MISS RATE", "LATENCY");
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "----", "-----", "----", "----", "---------", "-------");
}

int cmd_kernel(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza kernel";
        for (int i = 1; i < argc; ++i) {
            cmd += " ";
            cmd += argv[i];
        }
        return std::system(cmd.c_str());
    }

    bool watch_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--watch") == 0 || std::strcmp(argv[i], "-w") == 0) {
            watch_mode = true;
        }
    }

#ifndef CADENZA_TEST_MODE
    int fd = connect_local();
    if (fd < 0) {
        std::fprintf(stderr, "Error: cannot connect to cadenza daemon\n");
        return 1;
    }

    std::string msg = watch_mode ? "KERNEL WATCH\n" : "KERNEL SNAPSHOT\n";
    ::write(fd, msg.data(), msg.size());

    print_header();

    char buf[8192]{};
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        std::printf("%s", buf);
        if (!watch_mode) break;
    }
    ::close(fd);
#else
    // Test mode: display static example table
    print_header();
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "safety_monitor", "SAFETY", "CORE_2", "2.1%", "0.00%", "0.4ms");
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "walk_controller", "CONTROL", "CORE_1", "18.3%", "0.02%", "3.1ms");
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "imu_reader", "SENSOR", "CORE_1", "5.7%", "0.00%", "1.2ms");
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "terrain_classifier", "INFERENCE", "SHARED", "32.4%", "1.20%", "38.5ms");
    std::printf("%-20s %-12s %-8s %-12s %-12s %-10s\n",
                "fleet_sync", "COMMS", "SHARED", "1.0%", "0.00%", "12.0ms");
#endif

    return 0;
}

} // namespace cadenza::cli
