// Minimal JSON DOM parser for config.json, tokenizer.json and safetensors
// headers. Parses into one malloc'd node array; strings are unescaped copies.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jnode {
    jtype type;
    int n;        // ARR/OBJ: child count; STR: byte length
    int first;    // ARR/OBJ: index of first child (children are linked by `next`)
    int next;     // next sibling, -1 at the end
    const char *key; int key_len; // set on object members
    const char *s;                // STR: unescaped bytes (NUL-terminated)
    double num;                   // NUM, BOOL (0/1)
} jnode;

typedef struct {
    jnode *nodes;
    int n_nodes, cap;
    char *strs; size_t strs_len, strs_cap; // arena for unescaped strings
} json;

// Parse len bytes; returns NULL (with a message on stderr) on syntax errors.
json *json_parse(const char *text, size_t len);
json *json_load_file(const char *path);
void json_free(json *j);

static inline const jnode *json_root(const json *j) { return &j->nodes[0]; }
const jnode *json_get(const json *j, const jnode *obj, const char *key); // NULL if absent
const jnode *json_child(const json *j, const jnode *n, int i);           // i-th array/object child
static inline const jnode *json_next(const json *j, const jnode *n) { return n->next >= 0 ? &j->nodes[n->next] : NULL; }
static inline const jnode *json_first(const json *j, const jnode *n) { return n->first >= 0 ? &j->nodes[n->first] : NULL; }
double json_num(const json *j, const jnode *obj, const char *key, double def);
bool json_bool(const json *j, const jnode *obj, const char *key, bool def);
// Compact JSON text of a node (malloc'd).
char *json_dump(const json *j, const jnode *n);
const char *json_str(const json *j, const jnode *obj, const char *key); // NULL if absent / not a string
