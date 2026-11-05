#!/usr/bin/env python3
"""Sample a server and its descendants until the root exits (RSS is not GPU allocation)."""
import csv
from pathlib import Path
import subprocess
import sys
import time

pid = int(sys.argv[1])
output = Path(sys.argv[2])
with output.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['unix_seconds', 'processes', 'rss_sum_mb', 'cpu_percent_sum'])
    while True:
        rows = []
        for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,%cpu='], text=True).splitlines():
            fields = line.split()
            if len(fields) == 4:
                rows.append((int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3])))
        if not any(row[0] == pid for row in rows):
            break
        family = {pid}
        while True:
            expanded = family | {r[0] for r in rows if r[1] in family}
            if expanded == family:
                break
            family = expanded
        selected = [r for r in rows if r[0] in family]
        writer.writerow([time.time(), len(selected), sum(r[2] for r in selected)/1024, sum(r[3] for r in selected)])
        stream.flush()
        time.sleep(1)
