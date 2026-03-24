// cadenza shell --target <host>
// Opens an SSH session to the target host with the Cadenza environment
// pre-configured (PATH, LD_LIBRARY_PATH, CADENZA_ROOT).

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);

int cmd_shell(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (target.empty()) {
        std::fprintf(stderr, "Error: --target <host> is required for shell\n");
        std::fprintf(stderr, "Usage: cadenza shell --target <host>\n");
        return 1;
    }

    std::string ssh_key;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ssh-key") == 0 && i + 1 < argc) {
            ssh_key = argv[++i];
        }
    }

    // Build SSH command with Cadenza environment injected
    std::string ssh_cmd = "ssh -t";
    ssh_cmd += " -o StrictHostKeyChecking=accept-new";
    if (!ssh_key.empty()) {
        ssh_cmd += " -i " + ssh_key;
    }
    ssh_cmd += " " + target;

    // Remote command: source cadenza env then drop into interactive shell
    ssh_cmd += " 'export CADENZA_ROOT=/opt/cadenza"
               " && export PATH=$CADENZA_ROOT/bin:$PATH"
               " && export LD_LIBRARY_PATH=$CADENZA_ROOT/lib:$LD_LIBRARY_PATH"
               " && echo \"Cadenza shell on $(hostname)\""
               " && exec bash --login'";

    std::printf("Connecting to %s...\n", target.c_str());

#ifndef CADENZA_TEST_MODE
    int rc = std::system(ssh_cmd.c_str());
    return rc;
#else
    std::printf("[test mode] %s\n", ssh_cmd.c_str());
    return 0;
#endif
}

} // namespace cadenza::cli
