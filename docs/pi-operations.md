# Raspberry Pi operation and recovery

## Everyday use

Start: connect the Pi's normal power supply without holding the power button.
For hotspot access, enable Allow Others to Join and Maximize Compatibility on
the iPhone, and keep its Personal Hotspot screen open while the Pi connects.
The saved hotspot SSID is `Michael's stuff` (straight apostrophe).

Connect from the Mac with `ssh mli5@pi5`, or
`ssh mli5@100.85.175.6` through Tailscale. Campus client isolation can prevent
direct local-network access. Allow a couple of minutes for startup; check
`tailscale status` before making one SSH attempt.

Stop: briefly press the Pi's power button once. Raspberry Pi OS Lite uses this
to request a clean shutdown. Wait for shutdown to complete (normally tens of
seconds, activity stops and the power indicator returns to standby), then
unplug the power cable. Do not hold the button for routine shutdown: that forces
power off. A green LED or a quiet fan alone does not prove Linux is healthy.

Alternatively, run `ssh mli5@pi5 'sudo poweroff'`, wait for shutdown, then unplug.
If shutdown does not finish, investigate rather than repeatedly cutting power.

## USB recovery

Use recovery when normal startup or SSH fails; it is separate from normal use.
The existing Mac tools are in `~/code.tmp/usbboot`:

```sh
cd ~/code.tmp/usbboot
./rpiboot -d mass-storage-gadget64
```

With the Pi disconnected from power, hold its power button while connecting
its USB-C port to the Mac using a data cable; release the button once connected.
The Mac should first see BCM2712 Boot, then a recovery USB device and a serial
console under `/dev/cu.usbmodem*`. This boots a temporary recovery OS in RAM,
not the installed OS.

Before disk work, identify the device afresh using `diskutil list external`.
Never assume a disk number stays the same. Never initialize or erase a disk in
response to a macOS prompt.

**Only one system may mount a given partition at a time.** Unmount it on the
Mac before mounting it in the recovery console, and vice versa. Do not edit
the boot partition simultaneously from macOS and recovery Linux. Use
`ro,noload` for initial inspection of the ext4 root partition to avoid journal
replay. Back up files before repair; run filesystem checks on unmounted volumes.

To finish recovery: sync and unmount every NVMe partition in recovery Linux,
unmount/eject the exported disk on the Mac, then power off recovery Linux. Once
shutdown completes, disconnect the cable. For normal use, reconnect the normal
power supply without holding the button.

## Review on 2026-11-02

- Confirmed boot-partition corruption: `cmdline.txt` and its backup referenced
  clusters marked free; their contents were not valid boot arguments. This is
  a concrete startup fault. The cause and timing of corruption are unproven.
- Backed up 433 readable boot files and the damaged command-line bytes on the
  Mac under `~/.local/state/pi-recovery/2026-11-02/`.
- Repaired the FAT filesystem with `diskutil repairVolume`, recreated
  `cmdline.txt`, and passed a subsequent `diskutil verifyVolume` check.
- Verified root partition identity: `PARTUUID=8d963024-02`.
- The unmounted ext4 root filesystem passed `e2fsck -fn` (exit 0).
- Archived the experimental Stanford registration service/script and the
  ping-based network reboot watchdog under `/root/pi-recovery-20261102/` on
  the Pi, and removed their startup links. They are disabled. Normal
  NetworkManager autoconnect remains responsible for saved Wi-Fi networks.
- eduroam remains first priority, limited to 2.4 GHz; hotspot remains saved as
  fallback; unregistered Stanford remains excluded from autoconnect.
- Retained the previous boot parameters `pcie_aspm=off` and
  `nvme_core.default_ps_max_latency_us=0`. They were proposed for a suspected
  NVMe dropout; that original diagnosis remains unconfirmed.
- Stopped stale Mac connection-polling loops. No system upgrades performed.
- At the end of the session, verified the repaired boot line from recovery,
  unmounted both NVMe partitions, ejected the Mac disk, and issued
  `systemctl poweroff`. The recovery USB device then disconnected.

### Ethernet deployment, 16:32 PST

The room's CAT6 jack now provides a working Ethernet connection. Stanford's
Residential Network Tool record `/devices/7698` is Active for wired MAC
`88:a2:9e:7a:45:4a`. The user restarted after the indicated activation window.
Normal boot and SSH through Tailscale then succeeded; NetworkManager reports
`Wired connection 1` on `eth0`. Wi-Fi registration remains unresolved and is not
required for this wired setup.

Source commit `83f416f40ea017c12b7cff664577cbb05692917f` is deployed to
`~/fleetllm` on the Pi. `make NO_METAL=1 -j4` succeeded. The model is
`~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807,694,368 bytes).

To run a prompt from the Mac:

```sh
ssh mli5@pi5
cd ~/fleetllm
./build/fleet generate --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --prompt "What is the capital of France? Answer in one sentence." \
  --n 64 --temp 0 --budget-gb 8 --kv-gb 2
```

First output was the correct Paris answer, using NEON dot-product kernels and
four CPU threads. The batched-vs-sequential check passed 8/8 tasks at 64 generated
tokens (49.06 s sequential, 20.10 s batched). Build and test logs are in the Pi's
home directory: `fleetllm-build.log`, `pi-smoke.log`, `pi-correctness.log`.

At a sample under load, temperature was 49.9°C, fan tachometer 1,741 RPM, and
`vcgencmd get_throttled` returned `0x0`. The fan can stop when cool and can be
quiet at low speed. The Pi clock was corrected from the Mac before downloading
the model; NTP was enabled but had not synchronized.

This establishes successful boot, wired access, build, and inference. It does
not establish long-term hardware reliability. Keep the working wired baseline
before attempting further network experiments or system upgrades.

### Glimmer one-agent setup

The current model is `~/models/Muse-Glimmer-30B-Q2_K.gguf` (11.04 GB).
It replaces the earlier Unsloth UD-Q2_K_XL file. Native Q2_K and Q3_K CPU
kernels are deployed; source weights are not repeatedly converted into floats.
The current one-agent tests and commands are in
[`results/pi5/2026-11-02/glimmer-q2k`](../results/pi5/2026-11-02/glimmer-q2k/README.md).

On the Pi, run:

```sh
~/run-glimmer "Explain how a rainbow forms in a few sentences."
~/run-glimmer "What is the capital of France?" --n 16
```

`~/run-glimmer` is `scripts/pi-glimmer.sh`: CPU, four threads, FP16 KV,
250 MB target KV, row budget 64, no source weight pinning, greedy sampling,
64 output tokens by default. Optional Fleet flags follow the quoted prompt.
It is a one-request command and exits when generation finishes.

Two plain four-thread trials reached 0.93 tokens/sec; 80-token prompt processing
took about 31 seconds. DFlash with 1/2/3 proposals reached 0.98/0.99/0.82 tok/s
on the same prompt, with ~2 GB extra peak RSS and additional startup time.
Plain decoding is the recommended short-request default. These short trials do
not establish coding accuracy or long-context/multi-agent capacity.

The drafter remains available for explicit experiments:

```sh
~/run-glimmer "Explain how a rainbow forms." \
  --draft "$HOME/models/glimmer-dflash" --draft-kv-gb 0.064 \
  --spec-k 2 --spec-rows 3
```

Historical first CPU DFlash smoke-test results (before the packed Q2/Q3 kernels)
remain in [`glimmer-dflash`](../results/pi5/2026-11-02/glimmer-dflash/README.md).

### Current demo presets after the optimization sweep

The [optimization sweep](../results/pi5/2026-11-02/optimization-sweep/README.md)
supersedes the short-request recommendation above. Both original Glimmer and
the optional official Qwen3-8B Q4_K_M model remain installed.

```sh
~/run-demo qwen "Explain why rainbows have different colors, in two sentences."
~/run-demo glimmer "Your editing instruction followed by the current file"
~/serve-demo qwen
```

Qwen ordinary generation is about 2.25 tok/s. Glimmer remains about 0.96 on fresh
prose; verified prompt lookup reached 2.31 on the file edit (Qwen: 7.27). These
are short one-agent measurements, not a multi-agent capacity guarantee. No
pruning or coarser activation precision is enabled in these presets.

`run-demo` exits after one response. `serve-demo` keeps the model loaded at
`127.0.0.1:8090`, serves direct-answer chat requests, and allows one active
request. Stop it with Ctrl-C before a routine shutdown. To use it from the Mac,
open a second terminal and run:

```sh
ssh -L 8090:127.0.0.1:8090 mli5@100.85.175.6
```

Then clients on the Mac can use `http://127.0.0.1:8090/v1/chat/completions`
with model `pi-demo`. Launchers accept explicit Fleet flags after the quoted
prompt, or after the model choice for the server. No server starts on boot.

Power-button reference:
https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-button
