// cadenza rl — RL Adaptation Kernel (Cadenza Pro)

#include "commands.h"
#include <cstdio>

namespace cadenza::cli {

int cmd_rl(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("cadenza rl — RL Adaptation Kernel\n\n");
    std::printf("  The RLAK learns real-time joint corrections on top of your\n");
    std::printf("  action library, making your robot improve from live experience.\n\n");
    std::printf("  This feature is available in Cadenza Pro.\n");
    std::printf("  Learn more: https://cadenza.dev/pro\n");
    return 0;
}

} // namespace cadenza::cli
