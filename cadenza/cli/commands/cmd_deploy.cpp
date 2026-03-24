// cadenza deploy --target <host>
// Deploys the current build to a remote Cadenza host via rsync --checksum,
// then signals the daemon to hot-reload changed modules (no reboot).

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);
extern std::string ssh_prefix(const std::string& host);
extern int connect_local();

int cmd_deploy(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (target.empty()) {
        std::fprintf(stderr, "Error: --target <host> is required for deploy\n");
        std::fprintf(stderr, "Usage: cadenza deploy --target <host>\n");
        return 1;
    }

    // Parse optional flags
    bool dry_run = false;
    std::string ssh_key;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dry-run") == 0 || std::strcmp(argv[i], "-n") == 0) {
            dry_run = true;
        } else if (std::strcmp(argv[i], "--ssh-key") == 0 && i + 1 < argc) {
            ssh_key = argv[++i];
        }
    }

    std::printf("Deploying to %s...\n", target.c_str());

    // Build rsync command: checksum-based diff, compressed transfer
    std::string rsync_cmd = "rsync -az --checksum --delete --progress";
    if (dry_run) rsync_cmd += " --dry-run";
    if (!ssh_key.empty()) {
        rsync_cmd += " -e 'ssh -i " + ssh_key + "'";
    }
    rsync_cmd += " build/ " + target + ":/opt/cadenza/";

    std::printf("Syncing build artifacts...\n");

#ifndef CADENZA_TEST_MODE
    int rc = std::system(rsync_cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "Error: rsync failed (exit %d)\n", rc);
        return 1;
    }

    if (!dry_run) {
        // Signal the remote daemon to hot-reload
        std::printf("Signalling daemon to reload...\n");
        std::string reload_cmd = ssh_prefix(target) + "cadenza _reload";
        rc = std::system(reload_cmd.c_str());
        if (rc != 0) {
            std::fprintf(stderr, "Warning: reload signal failed — "
                         "manual restart may be required\n");
        }
    }
#else
    std::printf("[test mode] %s\n", rsync_cmd.c_str());
    if (!dry_run) {
        std::printf("[test mode] reload signal to %s\n", target.c_str());
    }
#endif

    std::printf("Deploy complete.\n");
    return 0;
}

} // namespace cadenza::cli
