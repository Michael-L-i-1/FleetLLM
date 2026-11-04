# Glimmer streaming handoff compatibility

This local plugin changes one predicate in the installed MuseGlimmer reasoning
parser. It does not change weights, inference, batching, prefix caching, prompts,
tool schemas, or the tool parser. The global installation is left unchanged.

Use both flags when launching the comparison backend:

```sh
--reasoning-parser-plugin /absolute/path/to/integrations/fleetpi/compat/glimmer_handoff.py \
--reasoning-parser fleetpi_glimmer_handoff
```

Continue using `--enable-auto-tool-choice --tool-call-parser muse_glimmer`.

The original streaming predicate decodes token IDs, while the preceding
reasoning extractor reads delivered text. Those clocks can differ because the
detokenizer buffers text. `DelegatingParser.parse_delta` permanently replaces
its accumulated text with the extractor's handoff on transition. If token IDs
trigger transition before that handoff exists, the tool prefix is discarded.
The override transitions only after the inherited extractor actually supplies
the framed tool-channel prefix. Prompt/nonstream reasoning detection is unchanged.

The plugin verifies SHA-256 of all three installed parser sources on import.
An updated installation must be reviewed and replayed before changing these
guards. The replay executes their actual Python AST with lightweight protocol
and tokenizer stubs: it loads no model, torch, MLX, or GPU runtime.

```sh
python3 integrations/fleetpi/compat/replay_glimmer_handoff.py \
  --vllm-root /path/to/site-packages/vllm
```

The original single-agent attempt must remain preserved. Raw capture of its
first response (4,253 prompt tokens, 103 output tokens) establishes a **model
tool-format failure**, not this handoff race: its invoke starts
`<atem:invoke name="ls<|message|>` with no closing quote or `>`. The embedded
bare message token starts a content channel, leaving the prior tool channel
without a complete invoke. Both original and patched predicates produce zero
executable calls and the same 203-character parameter/closing-tag tail, in
whole-response and character-by-character replay. Do not infer an intended
tool name or repair the invalid call. The externally interrupted attempt has
no completed time-to-success measurement; its failed early calls are quality
evidence, not evidence of single-agent inference speed.

A synthetic replay independently proves the handoff framing defect. In addition,
the installed reasoning parser also mishandles a missing reasoning `<|eom|>`
followed by a bare tool header when the whole block arrives together; this
predicate-only plugin does not repair that separate issue.

To also replay the captured malformed response, append:

```sh
--raw-response results/fleetpi/setup/single-request-capture/raw-completion.json
```

Use the same plugin for both vLLM comparison arms. A previously successful
eight-agent run with the original parser is historical evidence, not a fully
matched measurement against a newly patched single-agent arm.
