# Minimal FleetPi presentation

Designed with Claude Opus 5.5, then simplified per user feedback: large site captures, small bold labels and quiet metrics. Rendered and verified locally.

[Watch the presentation](quiet-video/comparison.mp4) · [Poster](quiet-video/poster.png) · [Original technical recording](../video/comparison.mp4)

The presentation shows the actual home-page screenshots from the frozen three-mode comparison. Large figures are measured whole-run workflow averages, including tools/checks: FleetPi8 112.7, vLLM8 38.0, vLLM1 14.2 output tokens/s. They are static summary metrics, not live GPU decode rates. A shared real-time clock advances equally at40×; each lane freezes at its actual end. The single agent's failed early milestones remain footnoted. All source image hashes, manifest positions and metric inputs are recorded in quiet-video/provenance.json and validated in validation.json.

The source benchmark, output websites, timings and original recording are unchanged. This is a presentation-only restyle. The original folder of recorded page images must be present to regenerate it (retained locally and in the separately hashed capture archive).

```sh
node demos/agent-task-bench/visual/compose-presentation.mjs \
  results/fleetpi/presentation/config.json results/fleetpi/presentation/quiet-video
```

Use --stills-only for quick layout QA. Output is1920×720,30fps,~28.8seconds.

## Terminal demo

The cmux workspace `FleetPi · Live demo` has the real eight-agent dashboard above a compact guide. From the repository root:

```sh
./fleetpi                    # interactive Pi with Fleet
./fleetpi demo festival      # eight page agents and cumulative browser checks
```

Only one model server runs at a time. Ctrl-C once stops the owned demo and saves progress. Resume a case with --mode fleet8 --resume --output <run-directory>. `python3 integrations/fleetpi/scripts/demo-guide.py` prints the quick guide again. The launcher accepts current Node>=22.19 and automatically falls back to an installed supported Homebrew Node if the terminal's nvm runtime is older; FLEETPI_NODE can select an explicit supported executable.
