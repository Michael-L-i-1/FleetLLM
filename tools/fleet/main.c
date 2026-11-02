// `fleet`: git-style entry point. `fleet <subcommand> [args...]` execs the
// fleet-<subcommand> binary next to this one (build/fleet-serve, ...), else
// the first fleet-<subcommand> on PATH. Each subcommand is a tools/<name>.c.
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

static const struct { const char *name, *desc; } cmds[] = {
    { "serve",        "OpenAI-compatible HTTP server (streaming, tool calls, KV cap, swap)" },
    { "generate",     "generate from one prompt (smoke test, single-sequence speed)" },
    { "bench",        "concurrent-agent throughput (--share, --draft, --check, --check-spec)" },
    { "ppl",          "perplexity of a text file, to measure lossy options" },
    { "reuse-eval",   "quality of reusing a span's KV computed in another context" },
    { "tokcheck",     "token ids per sample, compared with HF by scripts/tokcheck.py" },
    { "dequant-dump", "dequantized tensor rows, compared with gguf-py by scripts/dequantcheck.py" },
};

static void usage(FILE *f) {
    fprintf(f, "FleetLLM: high-throughput inference for fleets of agents on one machine.\n\n"
               "usage: fleet <command> [args...]\n\ncommands:\n");
    for (size_t i = 0; i < sizeof cmds / sizeof cmds[0]; i++) fprintf(f, "  %-13s %s\n", cmds[i].name, cmds[i].desc);
}

// Directory of this executable (symlinks resolved), or "" if unknown.
static void self_dir(const char *argv0, char *dir, size_t n) {
    char raw[PATH_MAX] = "", real[PATH_MAX];
#ifdef __APPLE__
    uint32_t sz = sizeof raw;
    if (_NSGetExecutablePath(raw, &sz) != 0) raw[0] = 0;
#else
    ssize_t k = readlink("/proc/self/exe", raw, sizeof raw - 1);
    raw[k > 0 ? k : 0] = 0;
#endif
    if (!raw[0] && strchr(argv0, '/')) snprintf(raw, sizeof raw, "%s", argv0);
    const char *p = raw[0] && realpath(raw, real) ? real : raw;
    const char *slash = strrchr(p, '/');
    snprintf(dir, n, "%.*s", slash ? (int)(slash - p) : 0, p);
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(stdout);
        return 0;
    }
    char name[256], dir[PATH_MAX], path[PATH_MAX + 256];
    snprintf(name, sizeof name, "fleet-%s", argv[1]);
    for (char *c = name; *c; c++) if (*c == '_') *c = '-';   // reuse_eval == reuse-eval
    self_dir(argv[0], dir, sizeof dir);
    argv[1] = name;
    if (dir[0]) {
        snprintf(path, sizeof path, "%s/%s", dir, name);
        if (access(path, X_OK) == 0) {
            argv[1] = path;
            execv(path, argv + 1);
            fprintf(stderr, "fleet: %s: %s\n", path, strerror(errno));
            return 126;
        }
    }
    execvp(name, argv + 1);
    fprintf(stderr, "fleet: unknown command '%s' (%s not found)\n\n", name + 6, name);
    usage(stderr);
    return 127;
}
