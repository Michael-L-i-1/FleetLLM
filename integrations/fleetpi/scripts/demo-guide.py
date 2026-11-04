#!/usr/bin/env python3
"""A compact terminal guide for the actual FleetPi live demo."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
cyan='\033[38;2;129;222;196m' if sys.stdout.isatty() else ''
dim='\033[2m' if sys.stdout.isatty() else ''
bold='\033[1m' if sys.stdout.isatty() else ''
reset='\033[0m' if sys.stdout.isatty() else ''
print(f'{cyan}{bold}FleetPi  /  quick start{reset}\n')
print(f'{dim}{root}{reset}')
print('  ./fleetpi                         Open the interactive Pi coding agent')
print('  ./fleetpi demo festival           Run 8 page agents + browser checks')
print('  ./fleetpi demo festival --compare Run the three modes sequentially\n')
print(f'{bold}What the live panel shows{reset}')
print('  8 real Pi sessions → shared context → Fleet GPU admission → file tools')
print('  Model permits are released during tools; finished pages can repair early.')
print('  Metal/NAX four-bit kernels + verified n-gram drafts + shared KV reuse.\n')
print(f'{bold}Stop / resume{reset}')
print('  One engine at a time: stop the live demo before opening ./fleetpi.')
print('  Ctrl-C once saves progress and stops the owned engine.')
print('  Resume: ./fleetpi demo festival --mode fleet8 --resume --output <run-dir>')
if len(sys.argv)>1: print(f'  Live run: {sys.argv[1]}')
print(f'{dim}Per-row req tok/s includes request time; workflow output tok/s is output / total task time.{reset}')
