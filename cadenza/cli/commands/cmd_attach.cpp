// cadenza attach <package>[@version]
// Hot-loads a package into the running Cadenza OS without reboot.
// Sends an ATTACH request to the daemon via Unix socket (or SSH for remote).

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

int cmd_attach(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza attach";
        for (int i = 1; i < argc; ++i) {
            cmd += " ";
            cmd += argv[i];
        }
        return std::system(cmd.c_str());
    }

    if (argc < 2) {
        std::fprintf(stderr, "Usage: cadenza attach <package>[@version]\n");
        return 1;
    }

    std::string spec = argv[1];
    std::string package_name = spec;
    std::string version = "latest";

    auto at_pos = spec.find('@');
    if (at_pos != std::string::npos) {
        package_name = spec.substr(0, at_pos);
        version = spec.substr(at_pos + 1);
    }

    std::printf("Attaching package '%s' version '%s'...\n",
                package_name.c_str(), version.c_str());

#ifndef CADENZA_TEST_MODE
    int fd = connect_local();
    if (fd < 0) {
        std::fprintf(stderr, "Error: cannot connect to cadenza daemon at "
                     "/run/cadenza/cadenza.sock\n");
        std::fprintf(stderr, "Is the Cadenza OS running?\n");
        return 1;
    }

    // Protocol: "ATTACH <package> <version>\n"
    std::string msg = "ATTACH " + package_name + " " + version + "\n";
    ::write(fd, msg.data(), msg.size());

    char buf[4096]{};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);

    if (n > 0) {
        std::printf("%s", buf);
    }
#else
    std::printf("OK: package '%s@%s' attached (test mode)\n",
                package_name.c_str(), version.c_str());
#endif

    return 0;
}

} // namespace cadenza::cli
