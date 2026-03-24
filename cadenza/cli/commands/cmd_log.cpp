// cadenza log [--topic <t>|--kernel <k>|--aios]
// Streams JSON log lines from the Cadenza OS to the terminal.

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

int cmd_log(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza log";
        for (int i = 1; i < argc; ++i) {
            cmd += " ";
            cmd += argv[i];
        }
        return std::system(cmd.c_str());
    }

    // Parse filter flags
    std::string topic;
    std::string kernel;
    bool aios_mode = false;
    int tail_n = 0; // 0 = stream live

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--topic") == 0 && i + 1 < argc) {
            topic = argv[++i];
        } else if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel = argv[++i];
        } else if (std::strcmp(argv[i], "--aios") == 0) {
            aios_mode = true;
        } else if (std::strcmp(argv[i], "--tail") == 0 && i + 1 < argc) {
            tail_n = std::atoi(argv[++i]);
        }
    }

    // Build daemon request
    std::string daemon_msg = "LOG STREAM";
    if (!topic.empty()) {
        daemon_msg += " TOPIC " + topic;
    } else if (!kernel.empty()) {
        daemon_msg += " KERNEL " + kernel;
    } else if (aios_mode) {
        daemon_msg += " AIOS";
    }
    if (tail_n > 0) {
        daemon_msg += " TAIL " + std::to_string(tail_n);
    }
    daemon_msg += "\n";

#ifndef CADENZA_TEST_MODE
    int fd = connect_local();
    if (fd < 0) {
        std::fprintf(stderr, "Error: cannot connect to cadenza daemon\n");
        return 1;
    }

    ::write(fd, daemon_msg.data(), daemon_msg.size());

    // Stream output until connection closes or Ctrl-C
    char buf[8192]{};
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        std::printf("%s", buf);
        std::fflush(stdout);
    }
    ::close(fd);
#else
    std::string filter_desc = "all";
    if (!topic.empty()) filter_desc = "topic=" + topic;
    else if (!kernel.empty()) filter_desc = "kernel=" + kernel;
    else if (aios_mode) filter_desc = "aios";

    std::printf("[test mode] streaming logs (%s)\n", filter_desc.c_str());
    std::printf("{\"ts\":1711100000.123,\"kernel\":\"dsk\",\"topic\":\"dsk/idle\","
                "\"idle_ms\":82.3}\n");
    std::printf("{\"ts\":1711100000.223,\"kernel\":\"rlak\",\"topic\":\"rlak/reward\","
                "\"reward\":0.823,\"episode\":1247}\n");
    std::printf("{\"ts\":1711100000.323,\"kernel\":\"aios\",\"topic\":\"aios/decisions\","
                "\"rule\":4,\"action\":\"RLAK_TRAIN\"}\n");
#endif

    return 0;
}

} // namespace cadenza::cli
