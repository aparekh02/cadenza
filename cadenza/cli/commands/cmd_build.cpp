// cadenza build
// Runs cmake + make against Cadenza headers in the current project directory.

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);
extern std::string ssh_prefix(const std::string& host);

int cmd_build(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza build";
        for (int i = 1; i < argc; ++i) { cmd += " "; cmd += argv[i]; }
        return std::system(cmd.c_str());
    }

    struct stat st{};
    if (stat("cadenza.toml", &st) != 0) {
        std::fprintf(stderr, "Error: cadenza.toml not found. Run 'cadenza init' first.\n");
        return 1;
    }

    int jobs = 0;
    bool verbose = false;
    bool clean = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-j") == 0 && i + 1 < argc) jobs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (std::strcmp(argv[i], "--clean") == 0) clean = true;
    }

    if (clean) {
        std::printf("Cleaning build directory...\n");
        std::system("rm -rf build");
    }

    std::system("mkdir -p build");

    std::printf("Configuring...\n");
    std::string cmake_cmd = "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release";
    if (verbose) cmake_cmd += " --log-level=VERBOSE";

#ifndef CADENZA_TEST_MODE
    int rc = std::system(cmake_cmd.c_str());
    if (rc != 0) { std::fprintf(stderr, "Error: cmake configure failed\n"); return 1; }
#else
    std::printf("[test mode] %s\n", cmake_cmd.c_str());
#endif

    std::printf("Building...\n");
    std::string make_cmd = "cmake --build build";
    if (jobs > 0) make_cmd += " -j " + std::to_string(jobs);

#ifndef CADENZA_TEST_MODE
    int rc2 = std::system(make_cmd.c_str());
    if (rc2 != 0) { std::fprintf(stderr, "Error: build failed\n"); return 1; }
#else
    std::printf("[test mode] %s\n", make_cmd.c_str());
#endif

    std::printf("Build complete.\n");
    return 0;
}

} // namespace cadenza::cli
