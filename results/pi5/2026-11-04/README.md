# Raspberry Pi website demo — November 4, 2026

Eight agents used Muse Glimmer 30B Q2_K on the Raspberry Pi 5 CPU. The orchestration harness and browser recorder ran on the Mac. The run generated website sections but **did not pass** its final checks: **55/63** checks passed. It is a visual progress demo, not a successful completion-time benchmark.

| Metric | Recorded value |
|---|---:|
| Task wall | 16,882.6 s (4 h 41 m) |
| Output tokens | 9,944 |
| Output / task wall | 0.589 tok/s |
| Model calls | 25 |
| Failed calls | 5 |
| Tool errors | 6 |

- [Run summary](coffee-site-v2-fleet8/summary.json), [events](coffee-site-v2-fleet8/events.jsonl), and [calls](coffee-site-v2-fleet8/calls.jsonl).
- [Generated sections](coffee-site-v2-fleet8/workspace/sections/), [rendered page](coffee-site-v2-fleet8/capture/preview/index.html), and [final checker](coffee-site-v2-fleet8/checks/002/result.json).
- [Pi development timelapse](coffee-site-v2-fleet8/video/pi-website-timelapse.mp4), [recording metadata](coffee-site-v2-fleet8/video/timelapse.json), and [sampled browser frames](coffee-site-v2-fleet8/capture/frames/).
- [Separate menu-edit copy](coffee-site-menu-edit/): subsequent Claude edits with illustrated drink cards. This copy is **not** Pi-generated benchmark output and is excluded from the Pi score and timelapse.

Rates here include waiting, tools and errors. They are not directly comparable with standalone model decoding rates or the different Mac festival task.
