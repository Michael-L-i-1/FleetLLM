#!/usr/bin/env python3
"""pimon: live terminal dashboard for the Raspberry Pi (or any Linux box over SSH).

    python3 scripts/pimon.py            # watches mli5@pi5
    python3 scripts/pimon.py user@host  # any Linux host with python3

One SSH connection runs a small collector on the host that prints one JSON
line per second (CPU, temperature, fan, power flags, memory, disk, network,
busy processes, inference jobs and the ports they serve, the newest log in ~,
and recent system warnings). This side keeps a minute of history and redraws.
If the host drops off the network it says so and keeps retrying.
Keys: q quit, r reconnect now. Standard library only.
"""

import json, os, queue, re, select, shutil, signal, subprocess, sys, termios, threading, time, tty
from collections import Counter, deque

# pimon.py [user@host] [--gpqa-local RUN_DIR [--gpqa-before CHECKPOINT ...]]
#   --gpqa-local: show a GPQA run on this machine in the GPQA tab (e.g. a Mac run that
#   continues the Pi's); --gpqa-before: earlier phases' checkpoints, for the combined line.
_args = sys.argv[1:]; LOCAL = None; BEFORE = []; HOST = "mli5@pi5"
while _args:
    a_ = _args.pop(0)
    if a_ == "--gpqa-local": LOCAL = _args.pop(0)
    elif a_ == "--gpqa-before": BEFORE.append(_args.pop(0))
    else: HOST = a_
HIST = 60

# --------------------------------------------------------------- collector (runs on the host)
REMOTE = r'''
import glob, json, os, re, socket, subprocess, sys, time
HOME = os.path.expanduser("~"); ME = os.getpid(); TCK = os.sysconf("SC_CLK_TCK"); PG = os.sysconf("SC_PAGE_SIZE")
JOB = re.compile(r"fleet|llama|gpqa|bench|vllm|serve|eval", re.I)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LDEC = re.compile(r"(?<!prompt )eval time =.*?([\d.]+) tokens per second")
SERVE = re.compile(r"(\d+) active \| ([\d.]+) tok/s generated \| (\d+) served")
def rd(p, d=""):
    try:
        with open(p, errors="replace") as f: return f.read()
    except Exception: return d
def sh(cmd, t=3):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t).stdout
    except Exception: return ""
def cpu():
    o = {}
    for l in rd("/proc/stat").splitlines():
        if l.startswith("cpu"):
            p = l.split(); v = list(map(int, p[1:9])); o[p[0]] = (sum(v), v[3] + v[4])
    return o
def procs():
    d = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        s = rd("/proc/%s/stat" % pid)
        r = s.rfind(")")
        if r < 0: continue
        f = s[r + 2:].split()
        try: d[int(pid)] = (int(f[11]) + int(f[12]), int(f[19]), int(f[21]) * PG, s[s.find("(") + 1:r])
        except Exception: pass
    return d
def netdev():
    o = {}
    for l in rd("/proc/net/dev").splitlines()[2:]:
        n, v = l.split(":", 1); v = v.split(); o[n.strip()] = (int(v[0]), int(v[8]))
    return o
def disk():
    for l in rd("/proc/diskstats").splitlines():
        p = l.split()
        if len(p) > 9 and p[2] in ("nvme0n1", "mmcblk0", "sda"): return p[2], int(p[5]) * 512, int(p[9]) * 512
    return "", 0, 0
def tcp():
    lis, est = {}, {}
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        for l in rd(f).splitlines()[1:]:
            p = l.split(); port = int(p[1].rsplit(":", 1)[1], 16)
            if p[3] == "0A": lis[p[9]] = port
            elif p[3] == "01": est[port] = est.get(port, 0) + 1
    return lis, est
def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 16384)); b = f.read().decode("utf-8", "replace")
    except Exception: return []
    out = []
    for l in b.split("\n"):
        l = ANSI.sub("", l.split("\r")[-1] if l.strip("\r") else l).rstrip()
        out.append(l)
    while out and not out[-1]: out.pop()
    return out[-n:]
c0, p0, n0, d0, t0 = cpu(), procs(), netdev(), disk(), time.time()
slow, tick = {}, 0
while True:
    time.sleep(1)
    c1, p1, n1, d1, t1 = cpu(), procs(), netdev(), disk(), time.time(); dt = t1 - t0
    up = float(rd("/proc/uptime", "0").split()[0])
    cores = []
    for k in sorted(c1, key=lambda k: (len(k), k)):
        a, b = c1[k], c0.get(k, c1[k]); tot = a[0] - b[0]
        cores.append(round(100 * (1 - (a[1] - b[1]) / tot), 1) if tot > 0 else 0)
    mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in rd("/proc/meminfo").splitlines() if l.split()[1:2] and l.split()[1].isdigit()}
    st = os.statvfs("/")
    if tick % 5 == 0:
        slow["thr"] = sh("vcgencmd get_throttled").strip().split("=")[-1]
        ips = {}
        for l in sh("ip -4 -o addr").splitlines():
            p = l.split(); ips[p[1]] = p[3].split("/")[0]
        slow["ips"] = ips
        route = ""
        for l in rd("/proc/net/route").splitlines()[1:]:
            p = l.split()
            if p[1] == "00000000" and (not route): route = p[0]
        slow["route"] = route
        slow["ssid"] = sh("iwgetid -r").strip() if route == "wlan0" or "wlan0" in ips else ""
        lis, _ = tcp(); owners = {}
        for pid in p1:
            if pid == ME: continue
            try:
                for fd in os.listdir("/proc/%d/fd" % pid):
                    ln = os.readlink("/proc/%d/fd/%s" % (pid, fd))
                    if ln.startswith("socket:["):
                        ino = ln[8:-1]
                        if ino in lis: owners.setdefault(pid, set()).add(lis[ino])
            except Exception: pass
        slow["ports"] = {pid: sorted(v) for pid, v in owners.items()}
        logs = [f for f in glob.glob(HOME + "/*.log") + glob.glob(HOME + "/*/*.log") + glob.glob(HOME + "/*/results/**/*.log", recursive=True) if os.path.isfile(f)]
        logs.sort(key=os.path.getmtime, reverse=True)
        slow["logs"] = [(f.replace(HOME, "~"), os.path.getmtime(f)) for f in logs[:5]]
    if tick % 10 == 0:
        w = sh("sudo -n journalctl -p warning -n 5 -o short --no-pager -q 2>/dev/null || journalctl -p warning -n 5 -o short --no-pager -q", 5)
        slow["warn"] = [l[:300] for l in w.splitlines() if l.strip()][-5:]
    _, est = tcp()
    rows = []
    for pid, (tk, start, rss, comm) in p1.items():
        if pid == ME: continue
        prev = p0.get(pid)
        pc = 100.0 * (tk - prev[0]) / (TCK * dt) if prev and prev[1] == start else 0.0
        rows.append((pc, pid, rss, start, comm))
    rows.sort(reverse=True)
    def cmd(pid, comm):
        c = rd("/proc/%d/cmdline" % pid).replace("\0", " ").strip()
        return c or "[" + comm + "]"
    top, jobs = [], []
    for pc, pid, rss, start, comm in rows:
        c = cmd(pid, comm) if (len(top) < 10 or JOB.search(comm)) else ""
        e = {"pid": pid, "cpu": round(pc, 1), "rss": rss, "age": round(up - start / TCK), "cmd": c[:400] or comm}
        isjob = bool(JOB.search(c or comm)) and "sshd" not in comm and pid != ME and "pimon" not in c
        if isjob:
            ports = slow.get("ports", {}).get(pid, [])
            e["ports"] = [[p, est.get(p, 0)] for p in ports]; jobs.append(e)
        if len(top) < 10: top.append(e)
    serve = []
    for j in jobs:
        if not re.search(r"fleet-serve|llama-server|serve", j["cmd"]): continue
        try: log = os.readlink("/proc/%d/fd/1" % j["pid"])
        except Exception: continue
        if not log.startswith("/") or not os.path.isfile(log): continue
        lines = tail(log, 400)
        stats = [[int(m.group(1)), float(m.group(2)), int(m.group(3))] for m in (SERVE.search(l) for l in lines) if m]
        done = [l for l in lines if re.search(r"generated tokens|total time", l)]
        dec = [float(m.group(1)) for m in (LDEC.search(l) for l in lines) if m]
        serve.append({"pid": j["pid"], "log": log.replace(HOME, "~"), "stats": stats[-120:], "done": done[-6:], "dec": dec[-120:],
                      "port": (j.get("ports") or [[0, 0]])[0], "cmd": j["cmd"]})
    if tick % 3 == 0:
        runs = [d for d in glob.glob(HOME + "/gpqa-run*") if os.path.isfile(d + "/checkpoint/questions.csv")]
        g = max(runs, key=lambda d: os.path.getmtime(d + "/checkpoint/questions.csv")) if runs else HOME + "/gpqa-run"
        ck = g + "/checkpoint"
        if os.path.isfile(ck + "/questions.csv"):
            import csv
            def num(x):
                try: return float(x)
                except Exception: return 0.0
            try:
                with open(ck + "/questions.csv", newline="") as f: qs = list(csv.DictReader(f))
                rq = [[int(r["index"]), r["status"], num(r["started"]), num(r["updated"]), num(r["seconds"]),
                       int(num(r["completion_tokens"])), int(num(r["answer_characters"])),
                       int(num(r["reasoning_characters"])), int(num(r["attempts"])), r.get("finish_reason") or ""] for r in qs]
            except Exception: rq = (slow.get("gpqa") or {}).get("rows", [])
            try: stj = json.loads(rd(ck + "/state.json", "{}"))
            except Exception: stj = {}
            spid = rd(g + "/supervisor.pid").strip()
            try: cap = json.loads(rd(ck + "/config.json", "{}")).get("max_tokens", 2048)
            except Exception: cap = 2048
            slow["gpqa"] = {"rows": rq, "state": stj, "control": rd(ck + "/control", "run").strip(),
                            "sup": rd(g + "/supervisor.log").splitlines()[-5:],
                            "sup_alive": bool(spid) and os.path.exists("/proc/" + spid),
                            "run": os.path.basename(g), "cap": cap}
    newest = slow.get("logs", [[None]])
    out = {
        "t": t1, "host": socket.gethostname(), "up": up, "cores": cores,
        "freq": int(rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "0").strip() or 0),
        "load": rd("/proc/loadavg").split()[:3],
        "temp": int(rd("/sys/class/thermal/thermal_zone0/temp", "0").strip() or 0) / 1000,
        "fan": int((rd(glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input")[0]) if glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input") else "0").strip() or 0),
        "pwm": int((rd(glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/pwm1")[0]) if glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/pwm1") else "0").strip() or 0),
        "thr": slow.get("thr", ""), "mem": [mi.get("MemTotal", 0), mi.get("MemAvailable", 0)],
        "swap": [mi.get("SwapTotal", 0), mi.get("SwapFree", 0)],
        "fs": [st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize],
        "disk": [d1[0], (d1[1] - d0[1]) / dt, (d1[2] - d0[2]) / dt],
        "net": {k: [(v[0] - n0.get(k, v)[0]) / dt, (v[1] - n0.get(k, v)[1]) / dt] for k, v in n1.items() if k != "lo"},
        "ips": slow.get("ips", {}), "route": slow.get("route", ""), "ssid": slow.get("ssid", ""),
        "top": top, "jobs": jobs[:6], "logs": slow.get("logs", []),
        "tail": tail(os.path.expanduser(newest[0][0])) if newest and newest[0] and newest[0][0] else [],
        "warn": slow.get("warn", []), "serve": serve, "gpqa": slow.get("gpqa"),
    }
    print(json.dumps(out), flush=True)
    c0, p0, n0, d0, t0 = c1, p1, n1, d1, t1; tick += 1
'''

# --------------------------------------------------------------- terminal drawing
E = "\x1b["
def fg(n): return f"{E}38;5;{n}m"
R = E + "0m"; B = E + "1m"; DIM = E + "2m"
C_BORDER, C_TITLE, C_TEXT, C_MUTED = fg(239), fg(114), fg(252), fg(244)
C_OK, C_WARN, C_BAD, C_ACC, C_ACC2 = fg(114), fg(214), fg(203), fg(81), fg(177)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SPARK = "▁▂▃▄▅▆▇█"

def vlen(s): return len(ANSI.sub("", s))

def fit(s, w):
    """Pad or cut s to exactly w visible columns, keeping escape codes intact."""
    out, n, i = [], 0, 0
    while i < len(s) and n < w:
        m = ANSI.match(s, i)
        if m: out.append(m.group()); i = m.end(); continue
        out.append(s[i]); n += 1; i += 1
    if n == w and i < len(s) and vlen(s[i:]) > 0 and w > 1:
        # mark truncation
        j = len(out) - 1
        while j >= 0 and ANSI.fullmatch(out[j]): j -= 1
        if j >= 0: out[j] = "…"
    return "".join(out) + R + " " * (w - n)

def heat(p, warn=70, bad=90):
    return C_BAD if p >= bad else C_WARN if p >= warn else C_OK

def bar(p, w, color=None):
    p = max(0.0, min(100.0, p)); full = p * w / 100; k = int(full)
    part = " ▏▎▍▌▋▊▉"[int((full - k) * 8)] if k < w else ""
    c = color or heat(p)
    s = c + "█" * k + (part if part.strip() else "")
    return s + C_BORDER + "·" * (w - k - (1 if part.strip() else 0)) + R

def spark(vals, w, lo=None, hi=None, color=C_ACC):
    v = list(vals)[-w:]
    if not v: return C_MUTED + "·" * w + R
    lo = min(v) if lo is None else lo; hi = max(v) if hi is None else hi
    span = (hi - lo) or 1
    s = "".join(SPARK[max(0, min(7, int((x - lo) / span * 7.999)))] for x in v)
    return C_MUTED + "·" * (w - len(v)) + color + s + R

def human(b, rate=False):
    u = ["B", "K", "M", "G", "T"]; x = float(b); i = 0
    while x >= 1024 and i < 4: x /= 1024; i += 1
    s = f"{x:.0f}{u[i]}" if x >= 100 or i == 0 else f"{x:.1f}{u[i]}"
    return s + ("/s" if rate else "")

def dur(s):
    s = int(max(0, s))
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s // 60}m{s % 60:02d}s"
    if s < 86400: return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600:02d}h"

def panel(title, lines, w, h=None, accent=C_TITLE):
    inner = w - 4
    t = f"{C_BORDER}╭─ {accent}{B}{title}{R}{C_BORDER} "
    top = t + "─" * max(0, w - vlen(t) - 1) + "╮" + R
    body = [lines[i] if i < len(lines) else "" for i in range(h if h is not None else len(lines))]
    return [top] + [f"{C_BORDER}│{R} {fit(l, inner)} {C_BORDER}│{R}" for l in body] + [C_BORDER + "╰" + "─" * (w - 2) + "╯" + R]

def side(a, b, gap=1):
    h = max(len(a), len(b)); wa, wb = vlen(a[0]), vlen(b[0])
    def grow(p, w):
        if len(p) < h:  # stretch the panel body, keep its bottom border last
            filler = f"{C_BORDER}│{R}" + " " * (w - 2) + f"{C_BORDER}│{R}"
            p = p[:-1] + [filler] * (h - len(p)) + p[-1:]
        return p
    a, b = grow(a, wa), grow(b, wb)
    return [x + " " * gap + y for x, y in zip(a, b)]

THROTTLE = {0: "under-voltage", 1: "freq capped", 2: "throttled", 3: "soft temp limit"}

# --------------------------------------------------------------- state
class State:
    def __init__(self):
        self.d = None; self.last = 0.0; self.online = False; self.since = time.time()
        self.attempt = 0; self.err = ""; self.next_try = 0.0
        self.cpu = deque(maxlen=HIST); self.temp = deque(maxlen=HIST); self.rx = deque(maxlen=HIST)
        self.tx = deque(maxlen=HIST); self.io = deque(maxlen=HIST); self.mem = deque(maxlen=HIST)
        self.tab = 1; self.note = ""; self.note_t = 0.0; self.gpqa_local = None; self.gpqa_t = 0.0
    def feed(self, d):
        self.d = d; self.last = time.time()
        if not self.online: self.online = True; self.since = self.last
        self.attempt = 0; self.err = ""
        c = d["cores"]; self.cpu.append(c[0] if c else 0); self.temp.append(d["temp"])
        rt = d.get("route") or "eth0"; n = d["net"].get(rt, [0, 0])
        self.rx.append(n[0]); self.tx.append(n[1]); self.io.append(d["disk"][1] + d["disk"][2])
        t, a = d["mem"]; self.mem.append(100 * (t - a) / t if t else 0)

# --------------------------------------------------------------- render
def render(S, W, H):
    W = max(W, 60); L = []
    d = S.d; now = time.time()
    # header
    if S.online:
        dot = f"{C_OK}●{R}"; status = f"{C_OK}live{R}"
    else:
        dot = f"{C_BAD}●{R}"; status = f"{C_BAD}{B}OFFLINE{R}{C_BAD} {dur(now - S.since)}{R}"
    host = (d or {}).get("host", HOST)
    net = ""
    if d:
        rt = d.get("route") or "—"; ip = d["ips"].get(rt, "")
        kind = "wired" if rt.startswith("eth") else (f"wifi {d['ssid']}" if d.get("ssid") else "wifi")
        ts = f"{C_OK}tailscale {d['ips']['tailscale0']}{R}" if "tailscale0" in d["ips"] else f"{C_BAD}tailscale down{R}"
        net = f"{C_TEXT}{rt}{R} {C_MUTED}{kind} {ip}{R}  {ts}  {C_MUTED}up {dur(d['up'])}{R}"
    tabs = "".join(f" {E}7m{C_TITLE}{B} {n} {name} {R}" if S.tab == n else f" {C_MUTED}{n} {name}{R}" for n, name in ((1, "system"), (2, "gpqa")))
    left = f" {dot} {C_TITLE}{B}pimon{R} {C_MUTED}▸{R} {C_TEXT}{B}{host}{R}  {status} {tabs}   {net}"
    right = f"{C_MUTED}{time.strftime('%H:%M:%S')} {R}"
    L.append(fit(left, W - vlen(right)) + right)
    if not S.online:
        msg = S.err or "connecting…"
        wait = max(0, S.next_try - now)
        L.append(f" {C_BAD}{B}▲ host unreachable{R} {C_MUTED}— {msg[:W - 40]}  · retry {S.attempt} in {wait:.0f}s · last data {dur(now - S.last) + ' ago' if S.last else 'never'}{R}")
    else:
        L.append("")
    if not d and not (S.tab == 2 and S.gpqa_local):
        L += ["", f"   {C_MUTED}waiting for first sample from {HOST} …{R}"]
        return L + [""] * (H - len(L) - 1) + [footer(W)]

    if S.tab == 2:
        L = render_gpqa(S, W, L)[:H - 1]
        return L + [""] * (H - 1 - len(L)) + [footer(W, S)]
    two = W >= 110; cw = (W - 1) // 2 if two else W
    # CPU
    cores = d["cores"]; total = cores[0] if cores else 0
    bw = max(10, cw - 26)
    cl = [f"{C_MUTED}all {R}{bar(total, bw)} {heat(total)}{total:5.1f}%{R}"]
    for i, p in enumerate(cores[1:]):
        cl.append(f"{C_MUTED}c{i:<2} {R}{bar(p, bw)} {C_TEXT}{p:5.1f}%{R}")
    cl.append(f"{C_MUTED}60s {R}{spark(S.cpu, bw, 0, 100)}  {C_TEXT}{d['freq'] / 1e6:.2f}GHz{R}")
    cl.append(f"{C_MUTED}load {C_TEXT}{' '.join(d['load'])}{R}")
    # thermal / power
    t = d["temp"]; tc = C_BAD if t >= 80 else C_WARN if t >= 70 else C_OK
    try: thr = int(d["thr"], 16)
    except ValueError: thr = 0
    now_f = [v for k, v in THROTTLE.items() if thr & (1 << k)]
    past_f = [v for k, v in THROTTLE.items() if thr & (1 << (k + 16)) and not thr & (1 << k)]
    pw = (f"{C_BAD}{B}{', '.join(now_f)}{R}" if now_f else f"{C_OK}power ok{R}") + (f" {C_WARN}(since boot: {', '.join(past_f)}){R}" if past_f else "")
    sw = max(10, cw - 24)
    tl = [f"{C_MUTED}temp {R}{tc}{B}{t:5.1f}°C{R}  {spark(S.temp, sw, min(S.temp) - 1, max(S.temp) + 1, tc)}",
          f"{C_MUTED}fan  {R}{C_TEXT}{d['fan']:>5} rpm{R}  {bar(d['pwm'] / 2.55, sw - 7, C_ACC)} {C_MUTED}{d['pwm'] / 2.55:3.0f}%{R}",
          f"{C_MUTED}pwr  {R}{pw}",
          f"{C_MUTED}heat {R}{bar((t - 30) / 55 * 100, sw + 9, tc)}",
          f"{C_MUTED}arm  {R}{C_TEXT}{d['freq'] / 1e6:.2f} GHz{R}  {C_MUTED}load {' '.join(d['load'])}{R}"]
    # memory / disk
    mt, ma = d["mem"]; mu = 100 * (mt - ma) / mt if mt else 0
    ft, fa = d["fs"]; fu = 100 * (ft - fa) / ft if ft else 0
    st_, sf = d["swap"]
    mb = max(10, cw - 30)
    ml = [f"{C_MUTED}mem  {R}{bar(mu, mb)} {C_TEXT}{human(mt - ma)}/{human(mt)}{R}",
          f"{C_MUTED}swap {R}{bar(100 * (st_ - sf) / st_ if st_ else 0, mb)} {C_TEXT}{human(st_ - sf)}/{human(st_)}{R}",
          f"{C_MUTED}disk {R}{bar(fu, mb)} {C_TEXT}{human(ft - fa)}/{human(ft)}{R}",
          f"{C_MUTED}{d['disk'][0] or 'io':<5}{R}{C_TEXT}r {human(d['disk'][1], True):>8}  w {human(d['disk'][2], True):>8}{R}  {spark(S.io, max(8, cw - 37))}"]
    # network
    nl = []
    for k in sorted(d["net"], key=lambda k: (k != d.get("route"), k)):
        rx, tx = d["net"][k]; up_ = k in d["ips"]
        mark = f"{C_OK}▲{R}" if k == d.get("route") else (f"{C_ACC}●{R}" if up_ else f"{C_MUTED}○{R}")
        nl.append(f"{mark} {C_TEXT}{k:<11}{R}{C_MUTED}{d['ips'].get(k, '—'):<16}{R} {C_ACC}↓{human(rx, True):>8}{R} {C_ACC2}↑{human(tx, True):>8}{R}")
    nl.append(f"{C_MUTED}↓ {R}{spark(S.rx, max(8, cw - 8), color=C_ACC)}")
    nl.append(f"{C_MUTED}↑ {R}{spark(S.tx, max(8, cw - 8), color=C_ACC2)}")

    if two:
        L += side(panel("CPU", cl, cw), panel("THERMAL · POWER", tl, W - cw - 1))
        L += side(panel("MEMORY · DISK", ml, cw), panel("NETWORK", nl, W - cw - 1))
    else:
        L += panel("CPU", cl, W) + panel("THERMAL · POWER", tl, W) + panel("MEMORY · DISK", ml, W) + panel("NETWORK", nl, W)

    # inference servers: parsed from the server's own log
    for sv in d.get("serve") or []:
        st = sv["stats"]; port, conns = sv["port"]
        il = []
        if st:
            act, tps, served = st[-1]
            gw = max(10, W - 52)
            il.append(f"{C_MUTED}active {R}{C_ACC2}{B}{act:>3}{R}  {C_MUTED}gen {R}{C_OK}{B}{tps:6.1f} tok/s{R}  {C_MUTED}served {R}{C_TEXT}{served:<5}{R} {spark([x[1] for x in st], gw, 0, None, C_OK)}")
            il.append(f"{C_MUTED}in flight {R}{bar(min(100, act / max(1, max(x[0] for x in st)) * 100), 20, C_ACC2)}  {C_MUTED}port {R}{C_ACC}:{port}{R} {C_MUTED}{conns} open conns · peak {max(x[1] for x in st):.1f} tok/s{R}")
        else:
            dec = sv.get("dec") or []
            il.append(f"{C_MUTED}in flight {R}{C_ACC2}{B}{conns:>3}{R} {C_MUTED}open conns on {R}{C_ACC}:{port}{R}"
                      + (f"   {C_MUTED}decode/request {R}{C_OK}{B}{dec[-1]:5.1f} tok/s{R} {C_MUTED}avg {sum(dec) / len(dec):.1f} over {len(dec)}{R}  {spark(dec, max(10, W - 100), 0, None, C_OK)}" if dec else f"   {C_MUTED}waiting for finished requests{R}"))
        il += [f"{C_TEXT}{x}{R}" for x in sv["done"][-3:]]
        L += panel(f"INFERENCE · {os.path.basename(sv['cmd'].split()[0])} · {sv['log']}", il, W, accent=C_OK)

    # jobs (inference / eval processes) and top processes
    jl = []
    for j in d["jobs"]:
        ports = "  ".join(f"{C_ACC}:{p}{R}{C_MUTED} {c} conn{'s' if c != 1 else ''}{R}" for p, c in j.get("ports", []))
        jl.append(f"{C_ACC2}{B}▶{R} {C_TEXT}{j['pid']:>6}{R} {heat(j['cpu'] / 4)}{j['cpu']:6.1f}%{R} {C_MUTED}{human(j['rss']):>6} {dur(j['age']):>7}{R}  {C_TEXT}{j['cmd']}{R}  {ports}")
    if not jl:
        jl.append(f"{C_MUTED}no fleet / llama / eval jobs running{R}")
    shown = {j["pid"] for j in d["jobs"]}
    rows = [p for p in d["top"] if p["pid"] not in shown and not (p["cpu"] < 1 and re.match(r"sshd-session|python3 -u -$|timeout \d|\[", p["cmd"]))]
    if rows: jl.append(f"{C_MUTED}{'PID':>8} {'CPU':>7} {'RSS':>6} {'AGE':>7}  COMMAND{R}")
    for p in rows[:5]:
        jl.append(f"  {C_TEXT}{p['pid']:>6}{R} {heat(p['cpu'] / 4)}{p['cpu']:6.1f}%{R} {C_MUTED}{human(p['rss']):>6} {dur(p['age']):>7}{R}  {C_TEXT}{p['cmd']}{R}")
    L += panel("JOBS · PROCESSES", jl, W, accent=C_ACC2)

    # warnings + log tail fill the rest
    warn = d.get("warn") or []
    logs = d.get("logs") or []
    rest = H - len(L) - 1
    if warn and rest > 12:
        L += panel("SYSTEM WARNINGS", [f"{C_WARN}{w}{R}" for w in warn[-3:]], W, accent=C_WARN)
        rest = H - len(L) - 1
    if logs and rest >= 4:
        name, mt_ = logs[0]
        others = "  ".join(f"{os.path.basename(n)} {C_MUTED}{dur(now - m)}{R}" for n, m in logs[1:4])
        lh = rest - 2
        body = [f"{C_MUTED}also: {R}{C_TEXT}{others}{R}"] if others else []
        tail = d.get("tail") or []
        body += [f"{C_TEXT}{x}{R}" for x in tail[-(lh - len(body)):]] if lh > len(body) else []
        L += panel(f"LOG {name} · {dur(now - mt_)} ago", body, W, h=lh, accent=C_ACC)
    L = L[:H - 1]
    return L + [""] * (H - 1 - len(L)) + [footer(W, S)]

def footer(W, S=None):
    note = f"   {S.note}" if S and S.note and time.time() - S.note_t < 8 else ""
    return fit(f"{E}7m {B}1{R}{E}7m system  {B}2{R}{E}7m gpqa  {B}p{R}{E}7m pause/resume gpqa  {B}r{R}{E}7m reconnect  {B}q{R}{E}7m quit   {DIM}{HOST}{note} ", W)

# --------------------------------------------------------------- GPQA tab
GLYPH = {"correct": (C_OK, "■"), "wrong": (C_BAD, "■"), "capped": (C_WARN, "■"), "error": (C_BAD, "!"),
         "interrupted": (C_WARN, "×"), "waiting": (C_ACC, "◇"), "reasoning": (C_ACC, "◆"),
         "answering": (C_ACC2, "◆"), "grading": (C_ACC2, "◆"), "queued": (C_BORDER, "·")}
FLIGHT = ("waiting", "reasoning", "answering", "grading")
DONE = ("correct", "wrong", "capped")
CAP = 2048

def render_gpqa(S, W, L):
    d = S.d or {}; g = S.gpqa_local or d.get("gpqa"); now = time.time()
    if not g:
        return L + panel("GPQA DIAMOND", [f"{C_MUTED}no run at ~/gpqa-run on the Pi yet{R}"], W)
    rows = g["rows"]; n = len(rows) or 1; cap = g.get("cap", CAP); run = g.get("run", "gpqa-run")
    cnt = Counter(r[1] for r in rows)
    fin = sum(cnt[s] for s in DONE); fl = sum(cnt[s] for s in FLIGHT)
    st = g.get("state") or {}; status = st.get("status", "?"); ctl = g.get("control", "run")
    sc = {"running": C_OK, "completed": C_OK, "paused": C_WARN, "pausing": C_WARN, "ready": C_MUTED}.get(status, C_BAD)
    sup = f"{C_OK}supervisor ✓{R}" if g.get("sup_alive") else f"{C_MUTED}supervisor not running{R}"
    el = st.get("elapsed_seconds", 0)
    rate = fin / (el / 3600) if el > 300 and fin else 0
    eta = f"{C_MUTED}ETA {R}{C_TEXT}{dur((n - fin) / rate * 3600)}{R}" if rate and fin >= 3 else f"{C_MUTED}ETA after 3 questions{R}"
    acc = 100 * cnt["correct"] / fin if fin else 0
    ans = cnt["correct"] + cnt["wrong"]; acc2 = 100 * cnt["correct"] / ans if ans else 0
    eng = ""
    if g.get("engine"):
        act, tps, served = g["engine"][-1]
        eng = f"{C_MUTED}engine {R}{C_ACC2}{act} active{R} {C_MUTED}·{R} {C_OK}{B}{tps:.1f} tok/s{R} {spark([x[1] for x in g['engine']], 30, 0, None, C_OK)}"
    for sv in ([] if g.get("engine") else d.get("serve") or []):
        if sv["stats"]:
            act, tps, served = sv["stats"][-1]
            eng = f"{C_MUTED}engine {R}{C_ACC2}{act} active{R} {C_MUTED}·{R} {C_OK}{B}{tps:.1f} tok/s{R} {spark([x[1] for x in sv['stats']], 30, 0, None, C_OK)}"
    sl = [f"{sc}{B}● {status}{R}  {C_MUTED}control {R}{C_TEXT}{ctl}{R}  {sup}   {C_MUTED}elapsed {R}{C_TEXT}{dur(el)}{R}  "
          f"{C_MUTED}rate {R}{C_TEXT}{rate:.1f} q/h{R}  {eta}",
          f"{C_MUTED}done {R}{bar(100 * fin / n, max(10, W - 30), C_OK)} {C_TEXT}{B}{fin}/{n}{R} {C_MUTED}{100 * fin / n:4.1f}%{R}",
          f"{C_OK}■ correct {cnt['correct']}{R}   {C_BAD}■ wrong {cnt['wrong']}{R}   {C_WARN}■ hit {cap}-token cap {cnt['capped']}{R}   "
          f"{C_ACC}◆ in flight {fl}{R}   {C_MUTED}· queued {cnt['queued']}{R}"
          + (f"   {C_BAD}! error {cnt['error']}{R}" if cnt["error"] else "") + (f"   {C_WARN}× interrupted {cnt['interrupted']}{R}" if cnt["interrupted"] else ""),
          f"{C_MUTED}score {R}{C_OK}{B}{acc:.1f}%{R} {C_MUTED}({cnt['correct']}/{fin} finished; capped count as wrong)  ·  "
          f"{acc2:.1f}% of answered  ·  random guessing 25%{R}"]
    if eng: sl.append(eng)
    cs = g.get("combined")
    if cs:
        parts = "  ".join(f"{C_TEXT}{p['run']}{R} {C_MUTED}({p['host'].split(' (')[0]}, {p['max_tokens']} tok){R} {C_OK}{p['correct_used']}{R}{C_MUTED}/{p['answers_used']}{R}"
                          for p in cs["phases"])
        sl.append(f"{C_MUTED}all phases {R}{C_OK}{B}{cs['correct']}/{cs['questions']} = {100 * cs['score']:.1f}%{R}  {C_MUTED}still open {cs['capped'] + cs['unfinished']}{R}   {parts}")
    where = f" on {g['host'].split(' (')[0]}" if g.get("host") else ""
    L = L + panel(f"GPQA DIAMOND · Muse Glimmer 30B Q2_K · {run}{where} · max {cap} tokens", sl, W, accent=C_OK)
    # one cell per question
    per = 50 if W - 4 >= 6 + 2 * 50 else 25
    by = {r[0]: r for r in rows}
    gl = []
    for a in range(1, len(rows) + 1, per):
        cells = " ".join((lambda c: c[0] + c[1] + R)(GLYPH.get(by[i][1], (C_MUTED, "?"))) for i in range(a, min(a + per, len(rows) + 1)) if i in by)
        gl.append(f"{C_MUTED}{a:>4} {R}{cells}")
    gl.append(f"{C_MUTED}     ■ correct  ■ wrong  ■ cap  ◆ reasoning  ◆ answering  ◇ reading prompt  · queued{R}")
    L += panel("QUESTIONS", gl, W, accent=C_ACC)
    # in flight
    live = sorted((r for r in rows if r[1] in FLIGHT), key=lambda r: r[2])
    il = []
    for r in live:
        idx, stt, started, _, _, _, ach, rch, att = r[:9]
        idx = (g.get("orig") or {}).get(idx, idx)
        tok = (ach + rch) / 4
        c, gch = GLYPH[stt]
        il.append(f"{C_TEXT}Q{idx:<4}{R}{c}{gch} {stt:<9}{R} {C_MUTED}{dur(now - started) if started else '—':>7}{R}  "
                  f"{bar(min(100, 100 * tok / cap), 24, C_ACC)} {C_TEXT}~{tok:>5.0f}{R}{C_MUTED}/{cap} tok{R}"
                  + (f"  {C_WARN}attempt {att}{R}" if att > 1 else ""))
    if not il: il = [f"{C_MUTED}nothing in flight{R}"]
    L += panel(f"IN FLIGHT · {len(live)}", il, W, accent=C_ACC2)
    # supervisor log
    sup_lines = [f"{C_MUTED}{x}{R}" for x in (g.get("sup") or [])[-3:]] or [f"{C_MUTED}(no supervisor log yet){R}"]
    L += panel("SUPERVISOR", sup_lines, W, accent=C_MUTED)
    # recent results fill the rest
    done = sorted((r for r in rows if r[1] in DONE), key=lambda r: -r[3])
    rl = []
    for r in done[:30]:
        idx, stt, _, upd, secs, ctok = r[:6]
        idx = (g.get("orig") or {}).get(idx, idx)
        c, gch = GLYPH[stt]
        rl.append(f"{C_TEXT}Q{idx:<4}{R}{c}{gch} {stt:<8}{R} {C_MUTED}{dur(secs):>8}  {ctok:>5} tok   finished {dur(now - upd)} ago{R}")
    if not rl: rl = [f"{C_MUTED}no finished questions yet{R}"]
    return L + panel("RECENT RESULTS", rl, W, accent=C_OK)

SERVE_RE = re.compile(r"(\d+) active \| ([\d.]+) tok/s generated \| (\d+) served")

def read_local_gpqa(run_dir, before):
    import csv
    rd_ = os.path.abspath(run_dir); ck = os.path.join(rd_, "checkpoint")
    num = lambda x: float(x) if x not in ("", None) else 0.0
    with open(os.path.join(ck, "questions.csv"), newline="") as f:
        rows = [[int(r["index"]), r["status"], num(r["started"]), num(r["updated"]), num(r["seconds"]), int(num(r["completion_tokens"])),
                 int(num(r["answer_characters"])), int(num(r["reasoning_characters"])), int(num(r["attempts"])), r.get("finish_reason") or ""]
                for r in csv.DictReader(f)]
    def text(p, default=""):
        try: return open(p, errors="replace").read()
        except OSError: return default
    pid = text(os.path.join(rd_, "supervisor.pid")).strip(); alive = False
    if pid:
        try: os.kill(int(pid), 0); alive = True
        except (OSError, ValueError): pass
    log = text(os.path.join(rd_, "server.log"))[-60000:]
    engine = [[int(m.group(1)), float(m.group(2)), int(m.group(3))] for m in SERVE_RE.finditer(log)][-120:]
    info = json.loads(text(os.path.join(rd_, "run_info.json"), "{}") or "{}")
    try: state = json.loads(text(os.path.join(ck, "state.json"), "{}"))
    except ValueError: state = {}
    g = {"rows": rows, "state": state, "control": text(os.path.join(ck, "control"), "run").strip() or "run",
         "sup": text(os.path.join(rd_, "supervisor.log")).splitlines()[-5:], "sup_alive": alive,
         "run": os.path.basename(rd_), "cap": json.loads(text(os.path.join(ck, "config.json"), "{}") or "{}").get("max_tokens", CAP),
         "host": info.get("host", ""), "engine": engine}
    if before:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evals"))
        import gpqa_extend
        g["combined"] = gpqa_extend.combine(list(before) + [ck])[1]
        # show each question by its number in the first phase (the original 198), not its row here
        first = {gpqa_extend.key(e): i + 1 for i, e in enumerate(json.load(open(os.path.join(before[0], "examples.json"))))}
        here = json.load(open(os.path.join(ck, "examples.json")))
        g["orig"] = {i + 1: first.get(gpqa_extend.key(e), i + 1) for i, e in enumerate(here)}
    return g

def send_control(S, action):
    def go():
        try:
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HOST, "fleetllm/evals/pi_gpqa.sh", action],
                           capture_output=True, timeout=20)
            S.note = f"gpqa: sent {action}"
        except Exception as e:
            S.note = f"gpqa: {action} failed ({e})"
        S.note_t = time.time()
    S.note = f"gpqa: sending {action}…"; S.note_t = time.time()
    threading.Thread(target=go, daemon=True).start()

# --------------------------------------------------------------- ssh plumbing
def connect(S, q):
    cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=3",
           "-o", "ServerAliveCountMax=2", HOST, "python3", "-u", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(REMOTE); p.stdin.close()
    def pump():
        for line in p.stdout:
            try: q.put(("data", json.loads(line)))
            except ValueError: pass
        err = p.stderr.read().strip().splitlines()
        q.put(("dead", err[-1] if err else f"ssh exited ({p.wait()})"))
    threading.Thread(target=pump, daemon=True).start()
    return p

def main():
    if not sys.stdout.isatty():
        sys.exit("pimon needs a terminal")
    S, q = State(), queue.Queue()
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    def restore(*_):
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"{E}?25h{E}?1049l"); sys.stdout.flush()
    signal.signal(signal.SIGTERM, lambda *a: (restore(), sys.exit(0)))
    tty.setcbreak(fd)
    sys.stdout.write(f"{E}?1049h{E}?25l"); sys.stdout.flush()
    proc, drawn = None, 0.0
    try:
        while True:
            now = time.time()
            if proc is None and now >= S.next_try:
                S.attempt += 1; proc = connect(S, q)
            if S.online and now - S.last > 8:  # stalled link: tear down and retry
                S.online = False; S.since = S.last; S.err = "no data for 8s"
                proc.kill(); proc = None; S.next_try = now
            while True:
                try: kind, v = q.get_nowait()
                except queue.Empty: break
                if kind == "data": S.feed(v)
                else:
                    if S.online: S.since = time.time()
                    S.online = False; S.err = v; proc = None
                    S.next_try = time.time() + min(30, 2 * S.attempt)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = os.read(fd, 16).decode(errors="ignore").lower()
                if "q" in k: break
                if "1" in k: S.tab = 1
                if "2" in k: S.tab = 2
                if "p" in k and S.gpqa_local:
                    cf = os.path.join(LOCAL, "checkpoint", "control")
                    new = "run" if S.gpqa_local.get("control") in ("pause", "stop") else "pause"
                    open(cf, "w").write(new + "\n"); S.note, S.note_t = f"gpqa: {new}", time.time()
                elif "p" in k and S.d and S.d.get("gpqa"):
                    send_control(S, "run" if S.d["gpqa"].get("control") in ("pause", "stop") else "pause")
                if "r" in k:
                    if proc: proc.kill()
                    proc = None; S.next_try = 0; S.attempt = 0
            if LOCAL and time.time() - S.gpqa_t >= 3:
                try: S.gpqa_local = read_local_gpqa(LOCAL, BEFORE)
                except Exception as e: S.note, S.note_t = f"local gpqa: {e}", time.time()
                S.gpqa_t = time.time()
            if time.time() - drawn >= 0.5:
                W, H = shutil.get_terminal_size((120, 40))
                frame = render(S, W, H)
                sys.stdout.write(E + "H" + "\n".join(fit(l, W) for l in frame[:H]) + E + "J"); sys.stdout.flush()
                drawn = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        if proc: proc.kill()
        restore()

if __name__ == "__main__":
    main()
