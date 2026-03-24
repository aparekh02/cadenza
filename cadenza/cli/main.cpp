// cadenza CLI — main entry point
// Parses argv[1] as command name and dispatches to the appropriate handler.
// Global --target <host> support: if present, commands operate on a remote host.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "commands/commands.h"

#ifndef CADENZA_VERSION
#define CADENZA_VERSION "0.0.0-dev"
#endif

namespace cadenza::cli {

// ---------------------------------------------------------------------------
// Target resolution helper — used by every command
// ---------------------------------------------------------------------------

static constexpr const char* CADENZA_SOCK = "/run/cadenza/cadenza.sock";

// Scans argv for --target <host>. If found, removes the two tokens from the
// argument list (shifts remaining args) and returns the host string.
// Returns empty string when no --target is present (local mode).
std::string extract_target(int& argc, char** argv) {
    for (int i = 0; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--target") == 0) {
            std::string host = argv[i + 1];
            // Remove --target and <host> from argv
            for (int j = i; j < argc - 2; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            return host;
        }
    }
    return "";
}

// Attempt to connect to the local Cadenza daemon via Unix socket.
// Returns the fd on success, -1 on failure.
int connect_local() {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, CADENZA_SOCK, sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// Build an ssh command prefix for remote execution.
std::string ssh_prefix(const std::string& host) {
    return "ssh -o StrictHostKeyChecking=accept-new " + host + " ";
}

} // namespace cadenza::cli

// ---------------------------------------------------------------------------
// Command dispatch table
// ---------------------------------------------------------------------------

struct CommandEntry {
    const char* name;
    int (*handler)(int argc, char** argv);
    const char* brief;
};

static const CommandEntry commands[] = {
    {"init",    cadenza::cli::cmd_init,    "Scaffold a new Cadenza project"},
    {"attach",  cadenza::cli::cmd_attach,  "Hot-load a package (no reboot)"},
    {"detach",  cadenza::cli::cmd_detach,  "Hot-unload a package"},
    {"build",   cadenza::cli::cmd_build,   "Build with cmake + make"},
    {"deploy",  cadenza::cli::cmd_deploy,  "Deploy to target via rsync"},
    {"shell",   cadenza::cli::cmd_shell,   "SSH shell with Cadenza env"},
    {"kernel",  cadenza::cli::cmd_kernel,  "Show kernel task table"},
    {"rl",      cadenza::cli::cmd_rl,      "RL adaptation control"},
    {"sim",     cadenza::cli::cmd_sim,     "Simulation package control"},
    {"log",     cadenza::cli::cmd_log,     "Stream JSON log lines"},
    {"aios",    cadenza::cli::cmd_aios,    "AIOS governor status"},
    {"ai",      cadenza::cli::cmd_ai,      "AICore intelligence kernel"},
};

static constexpr int NUM_COMMANDS = sizeof(commands) / sizeof(commands[0]);

static void print_usage() {
    std::fprintf(stderr, "Cadenza OS CLI v%s\n\n", CADENZA_VERSION);
    std::fprintf(stderr, "Usage: cadenza [--target <host>] <command> [args...]\n\n");
    std::fprintf(stderr, "Commands:\n");
    for (int i = 0; i < NUM_COMMANDS; ++i) {
        std::fprintf(stderr, "  %-10s %s\n", commands[i].name, commands[i].brief);
    }
    std::fprintf(stderr, "\nGlobal flags:\n");
    std::fprintf(stderr, "  --target <host>   Execute on remote Cadenza host\n");
    std::fprintf(stderr, "  --version         Print version and exit\n");
    std::fprintf(stderr, "  --help            Show this help\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    // Handle top-level flags before command dispatch
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("cadenza %s\n", CADENZA_VERSION);
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    // Skip past global --target before finding command name
    int cmd_index = 1;
    if (std::strcmp(argv[1], "--target") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "Error: --target requires a host argument\n");
            return 1;
        }
        cmd_index = 3; // argv[1]=--target, argv[2]=host, argv[3]=command
    }

    if (cmd_index >= argc) {
        print_usage();
        return 1;
    }

    const char* cmd_name = argv[cmd_index];

    for (int i = 0; i < NUM_COMMANDS; ++i) {
        if (std::strcmp(cmd_name, commands[i].name) == 0) {
            // Pass the full argv (including --target) so commands can call
            // extract_target themselves. Shift argc/argv to start at command.
            int sub_argc = argc - cmd_index;
            char** sub_argv = argv + cmd_index;
            return commands[i].handler(sub_argc, sub_argv);
        }
    }

    std::fprintf(stderr, "cadenza: unknown command '%s'\n", cmd_name);
    std::fprintf(stderr, "Run 'cadenza --help' for available commands.\n");
    return 1;
}
