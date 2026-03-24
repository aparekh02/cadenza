// cadenza init <name> --robot <go1|g1>
// Scaffolds a new Cadenza project directory with cadenza.toml, src/, and packages/.

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);
extern std::string ssh_prefix(const std::string& host);

static void mkdirs(const std::string& path) {
#ifdef CADENZA_TEST_MODE
    std::printf("[test mode] mkdir -p %s\n", path.c_str());
#else
    std::string cmd = "mkdir -p " + path;
    std::system(cmd.c_str());
#endif
}

static void write_toml(const std::string& dir, const std::string& name,
                       const std::string& robot) {
    std::ofstream f(dir + "/cadenza.toml");
    f << "[project]\n"
      << "name    = \"" << name << "\"\n"
      << "robot   = \"" << robot << "\"\n"
      << "version = \"0.1.0\"\n\n"
      << "[kernels]\n"
      << "rlak    = true\n"
      << "hack    = true\n"
      << "aik     = true\n"
      << "aios    = true\n\n"
      << "[packages]\n\n"
      << "[rl]\n"
      << "delta_clip    = 0.12\n"
      << "train_trigger = \"on_idle\"\n\n"
      << "[deploy]\n"
      << "targets    = []\n"
      << "hot_reload = true\n\n"
      << "[aios]\n"
      << "fallback_mode  = \"static\"\n"
      << "log_decisions  = true\n";
}

int cmd_init(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza init";
        for (int i = 1; i < argc; ++i) { cmd += " "; cmd += argv[i]; }
        return std::system(cmd.c_str());
    }

    std::string name, robot;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--robot") == 0 && i + 1 < argc) {
            robot = argv[++i];
        } else if (argv[i][0] != '-') {
            name = argv[i];
        }
    }

    if (name.empty()) { std::fprintf(stderr, "Usage: cadenza init <name> --robot <go1|g1>\n"); return 1; }
    if (robot.empty()) { std::fprintf(stderr, "Error: --robot <go1|g1> is required\n"); return 1; }
    if (robot != "go1" && robot != "g1") {
        std::fprintf(stderr, "Error: robot must be 'go1' or 'g1'\n"); return 1;
    }

    struct stat st{};
    if (stat(name.c_str(), &st) == 0) {
        std::fprintf(stderr, "Error: directory '%s' already exists\n", name.c_str());
        return 1;
    }

    mkdirs(name + "/src");
    mkdirs(name + "/packages");
    mkdirs(name + "/tests");

    write_toml(name, name, robot);

    {
        std::ofstream f(name + "/src/main.cpp");
        f << "#include <cadenza_api.h>\n\nint main() {\n    return 0;\n}\n";
    }

    std::printf("Created project '%s' for robot '%s'\n", name.c_str(), robot.c_str());
    return 0;
}

} // namespace cadenza::cli
