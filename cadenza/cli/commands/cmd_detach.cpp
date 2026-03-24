// cadenza detach <package>
// Hot-unloads a package from the running Cadenza OS without reboot.

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

int cmd_detach(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza detach";
        for (int i = 1; i < argc; ++i) {
            cmd += " ";
            cmd += argv[i];
        }
        return std::system(cmd.c_str());
    }

    if (argc < 2) {
        std::fprintf(stderr, "Usage: cadenza detach <package>\n");
        return 1;
    }

    std::string package_name = argv[1];

    std::printf("Detaching package '%s'...\n", package_name.c_str());

#ifndef CADENZA_TEST_MODE
    int fd = connect_local();
    if (fd < 0) {
        std::fprintf(stderr, "Error: cannot connect to cadenza daemon at "
                     "/run/cadenza/cadenza.sock\n");
        std::fprintf(stderr, "Is the Cadenza OS running?\n");
        return 1;
    }

    std::string msg = "DETACH " + package_name + "\n";
    ::write(fd, msg.data(), msg.size());

    char buf[4096]{};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);

    if (n > 0) {
        std::printf("%s", buf);
    }
#else
    std::printf("OK: package '%s' detached (test mode)\n",
                package_name.c_str());
#endif

    return 0;
}

} // namespace cadenza::cli
