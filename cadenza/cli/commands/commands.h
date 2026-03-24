#ifndef CADENZA_CLI_COMMANDS_H
#define CADENZA_CLI_COMMANDS_H

namespace cadenza::cli {

int cmd_init(int argc, char** argv);
int cmd_attach(int argc, char** argv);
int cmd_detach(int argc, char** argv);
int cmd_build(int argc, char** argv);
int cmd_deploy(int argc, char** argv);
int cmd_shell(int argc, char** argv);
int cmd_kernel(int argc, char** argv);
int cmd_rl(int argc, char** argv);
int cmd_sim(int argc, char** argv);
int cmd_log(int argc, char** argv);
int cmd_aios(int argc, char** argv);
int cmd_ai(int argc, char** argv);

} // namespace cadenza::cli

#endif // CADENZA_CLI_COMMANDS_H
