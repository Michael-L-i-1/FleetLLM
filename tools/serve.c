// OpenAI-compatible server (see src/server.h).
//   fleet serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 [--port 8090] [--max-active 64]
//               [--draft glimmer-draft] [--reasoning high] [--max-tokens 4096] [--name ID]
#include "common.h"
#include "server.h"

int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    server_opts o = { 0 };
    o.host = arg_str(argc, argv, "--host", "127.0.0.1");
    o.port = (int)arg_num(argc, argv, "--port", 8090);
    o.model_id = arg_str(argc, argv, "--name", arg_str(argc, argv, "--model", "model"));
    o.default_max_tokens = (int)arg_num(argc, argv, "--max-tokens", 4096);
    o.reasoning = arg_str(argc, argv, "--reasoning", NULL);
    o.max_active = (int)arg_num(argc, argv, "--max-active", 64);
    o.admit = arg_flag(argc, argv, "--admit") || (arg_num(argc, argv, "--swap-gb", 0) > 0 && !arg_flag(argc, argv, "--no-admit"));
    return server_run(rt.e, &o);
}
