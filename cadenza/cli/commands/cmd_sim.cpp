// cadenza sim [start|stop|list] <package>
// Controls simulation packages.

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

static void print_sim_usage() {
    std::fprintf(stderr, "Usage: cadenza sim <subcommand> [<package>]\n\n");
    std::fprintf(stderr, "Subcommands:\n");
    std::fprintf(stderr, "  start <package>  Start a simulation package\n");
    std::fprintf(stderr, "  stop <package>   Stop a running simulation package\n");
    std::fprintf(stderr, "  list             List all simulation packages and status\n");
}

int cmd_sim(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza sim";
        for (int i = 1; i < argc; ++i) {
            cmd += " ";
            cmd += argv[i];
        }
        return std::system(cmd.c_str());
    }

    if (argc < 2) {
        print_sim_usage();
        return 1;
    }

    std::string subcmd = argv[1];
    std::string daemon_msg;

    if (subcmd == "start") {
        if (argc < 3) {
            std::fprintf(stderr, "Error: start requires a package name\n");
            return 1;
        }
        daemon_msg = "SIM START " + std::string(argv[2]) + "\n";
    } else if (subcmd == "stop") {
        if (argc < 3) {
            std::fprintf(stderr, "Error: stop requires a package name\n");
            return 1;
        }
        daemon_msg = "SIM STOP " + std::string(argv[2]) + "\n";
    } else if (subcmd == "list") {
        daemon_msg = "SIM LIST\n";
    } else {
        std::fprintf(stderr, "Error: unknown sim subcommand '%s'\n",
                     subcmd.c_str());
        print_sim_usage();
        return 1;
    }

#ifndef CADENZA_TEST_MODE
    int fd = connect_local();
    if (fd < 0) {
        std::fprintf(stderr, "Error: cannot connect to cadenza daemon\n");
        return 1;
    }

    ::write(fd, daemon_msg.data(), daemon_msg.size());

    char buf[8192]{};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);

    if (n > 0) {
        std::printf("%s", buf);
    }
#else
    if (subcmd == "start") {
        std::printf("Simulation package '%s' started\n", argv[2]);
    } else if (subcmd == "stop") {
        std::printf("Simulation package '%s' stopped\n", argv[2]);
    } else if (subcmd == "list") {
        std::printf("%-24s %-10s %-12s %-20s\n",
                    "PACKAGE", "STATUS", "TICK RATE", "TOPICS");
        std::printf("%-24s %-10s %-12s %-20s\n",
                    "sim:terrain-rough", "running", "50ms", "sensors/lidar");
        std::printf("%-24s %-10s %-12s %-20s\n",
                    "sim:imu-noise", "stopped", "10ms", "sensors/imu_external");
    }
#endif

    return 0;
}

} // namespace cadenza::cli
