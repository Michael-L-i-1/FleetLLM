# November 4 results and demos

| Experiment | Result | Evidence |
|---|---|---|
| Mac festival website, eight agents | FleetPi + Fleet: 339.4 s; Pi + vLLM-Metal: 1,029.3 s. Both pass all stages; 3.03× task speedup in this trial. | [Report](../fleetpi/README.md), [JSON](../fleetpi/summary.json), [video](../fleetpi/presentation/annotated-demo/comparison.mp4) |
| GPQA Diamond, Pi-only | 115/198 correct (58.1%); 35 output-capped questions count as wrong. | [Primary run](../pi5/2026-11-03/gpqa-diamond-full/gpqa.json) |
| GPQA Diamond, phased larger budgets | 130/198 correct (65.7%); 170 retained answers from Pi and 28 from Mac. | [Method and phases](../gpqa/2026-11-04-glimmer-q2k-phased/README.md), [per-question CSV](../gpqa/2026-11-04-glimmer-q2k-phased/questions.csv) |
| Pi coffee website, eight agents | Generated sections; failed final checks, 55/63. | [Report](../pi5/2026-11-04/README.md), [timelapse](../pi5/2026-11-04/coffee-site-v2-fleet8/video/pi-website-timelapse.mp4) |
| Earlier Mac four-worker site refactor | Historical comparison, distinct task and configuration. | [Report](../mac/2026-11-03/site-refactor/README.md) |

[Archive manifest](archive-manifest.json) records file sizes and SHA-256 hashes for the newly copied local evidence, including the earlier site-refactor harness snapshot. The current `demos/site-refactor` implementation remains unchanged. Runtime PID/lock files, dependency caches and intermediate video work directories are excluded; actual captures, response records, checkpoint CSVs and final videos are retained.

The Mac matched comparison is a complete-system result. Component-by-component contributions have not been isolated, and batching/prefix caching are also supported by the comparator. The phased GPQA result is not a uniform Pi-only run. These distinctions are preserved in the source reports.
