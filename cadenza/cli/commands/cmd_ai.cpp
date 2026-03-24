// cadenza ai [status|goal <text>|decide|history]
// Interface to the AICore multimodal intelligence kernel.

#include "commands.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cadenza::cli {

extern std::string extract_target(int& argc, char** argv);
extern std::string ssh_prefix(const std::string& host);

static void print_ai_usage() {
    std::printf("Usage: cadenza ai <subcommand>\n\n");
    std::printf("Subcommands:\n");
    std::printf("  status       AICore kernel health, model info, decision rate\n");
    std::printf("  goal <text>  Set a high-level goal for the robot\n");
    std::printf("  decide       Show the current action decision with reasoning\n");
    std::printf("  world        Show the fused world state\n");
    std::printf("  history      Last 10 decisions with reasoning\n");
}

int cmd_ai(int argc, char** argv) {
    std::string target = extract_target(argc, argv);

    if (!target.empty()) {
        std::string cmd = ssh_prefix(target) + "cadenza ai";
        for (int i = 1; i < argc; ++i) { cmd += " "; cmd += argv[i]; }
        return std::system(cmd.c_str());
    }

    if (argc < 2) {
        print_ai_usage();
        return 1;
    }

    const char* sub = argv[1];

#ifdef CADENZA_TEST_MODE
    if (std::strcmp(sub, "status") == 0) {
        std::printf("AICore Status\n");
        std::printf("  Model:           builtin_behavior_tree\n");
        std::printf("  Decisions:       0\n");
        std::printf("  Perception:      0 us\n");
        std::printf("  Reasoning:       0 us\n");
        std::printf("  Confidence:      1.00\n");
        std::printf("  SLM:             not connected\n");
    } else if (std::strcmp(sub, "goal") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "Usage: cadenza ai goal <text>\n");
            return 1;
        }
        std::string goal;
        for (int i = 2; i < argc; ++i) {
            if (i > 2) goal += " ";
            goal += argv[i];
        }
        std::printf("Goal set: \"%s\"\n", goal.c_str());
    } else if (std::strcmp(sub, "decide") == 0) {
        std::printf("Current Decision\n");
        std::printf("  Action:     stand\n");
        std::printf("  Speed:      1.0\n");
        std::printf("  Confidence: 1.00\n");
        std::printf("  Layer:      IDLE\n");
        std::printf("  Reasoning:  No goal, standing by\n");
    } else if (std::strcmp(sub, "world") == 0) {
        std::printf("World State\n");
        std::printf("  Body:    roll=0.00 pitch=0.00 yaw=0.00 height=0.28m\n");
        std::printf("  Terrain: flat (class=0) roughness=0.00 slope=0.00\n");
        std::printf("  Goal:    (none)\n");
        std::printf("  Confidence: 1.00\n");
    } else if (std::strcmp(sub, "history") == 0) {
        std::printf("Decision History (last 10)\n");
        std::printf("  (no decisions yet)\n");
    } else {
        print_ai_usage();
        return 1;
    }
#else
    // Connect to daemon and query AICore
    (void)sub;
    std::printf("[live mode] Connect to /run/cadenza/cadenza.sock for AICore data\n");
#endif

    return 0;
}

} // namespace cadenza::cli
