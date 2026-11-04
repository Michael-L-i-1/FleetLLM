#!/usr/bin/env python3
"""Sample owned process trees; Darwin footprint includes charged GPU/IOKit memory."""
import csv
import ctypes
from pathlib import Path
import struct
import subprocess
import sys
import time

pid=int(sys.argv[1]);output=Path(sys.argv[2])
try:
    proc=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    proc.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    proc.proc_pid_rusage.restype=ctypes.c_int
except OSError:proc=None


def footprint(process):
    if proc is None:return None
    # RUSAGE_INFO_V4 from Apple's sys/resource.h: UUID16, then uint64 fields.
    data=ctypes.create_string_buffer(1024)
    if proc.proc_pid_rusage(process,4,ctypes.byref(data))!=0:return None
    return struct.unpack_from('Q',data.raw,72)[0],struct.unpack_from('Q',data.raw,240)[0]


with output.open('w',newline='') as out:
    writer=csv.writer(out,lineterminator='\n')
    writer.writerow(['unix_seconds','processes','rss_sum_mb','cpu_percent_sum','physical_footprint_sum_mb','footprint_processes','largest_process_lifetime_peak_mb'])
    while True:
        rows=[]
        for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss=,%cpu='],text=True).splitlines():
            f=line.split()
            if len(f)==4:rows.append((int(f[0]),int(f[1]),int(f[2]),float(f[3])))
        if not any(r[0]==pid for r in rows):break
        family={pid}
        while True:
            expanded=family|{r[0] for r in rows if r[1] in family}
            if expanded==family:break
            family=expanded
        selected=[r for r in rows if r[0] in family]
        values=[x for r in selected if (x:=footprint(r[0])) is not None]
        writer.writerow([time.time(),len(selected),sum(r[2] for r in selected)/1024,sum(r[3] for r in selected),
                         sum(x[0] for x in values)/1048576 if values else '',len(values),
                         max((x[1] for x in values),default=0)/1048576])
        out.flush();time.sleep(1)
