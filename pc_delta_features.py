#!/usr/bin/env python3
"""
pc_delta_features.py
Parse trace.txt (I lines) -> compute PC deltas and windowed features.
Produces features.csv with one row per executed instruction (starting from when enough history exists).
"""

import re
import math
import csv
from collections import deque, Counter
import numpy as np

# ---------- Config ----------
TRACE_PATH = 'trace.txt'
OUT_CSV = 'pc_features.csv'
K = 8           # last-k deltas
WIN = 32        # rolling window for stats
NGRAM = 2       # n-gram size for delta subsequences

# ---------- Helpers ----------
I_RE = re.compile(r'^I\s+0x([0-9a-fA-F]+)')

def hex_to_int(s): return int(s, 16)

def entropy_of_list(lst):
    if not lst:
        return 0.0
    c = Counter(lst)
    total = sum(c.values())
    ent = 0.0
    for v in c.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent

def fft_lowfreq_fraction(arr, keep_frac=0.2):
    # arr: 1D numeric list, return fraction of energy in lowest keep_frac frequencies
    if len(arr) < 4:
        return 0.0
    a = np.array(arr, dtype=float)
    a = a - a.mean()
    spec = np.fft.rfft(a)
    power = np.abs(spec)**2
    cutoff = max(1, int(len(power) * keep_frac))
    low = power[:cutoff].sum()
    total = power.sum() + 1e-12
    return float(low / total)

# ---------- Parse I lines and compute deltas ----------
pcs = []
with open(TRACE_PATH, 'r') as f:
    for line in f:
        if line.startswith("I"):
            addr = line.split()[1].strip()
            if addr.startswith("Ox"):
                addr = "0x" + addr[2:]
            pcs.append(int(addr,16))

print(f"Collected {len(pcs)} PCs")

# compute deltas
deltas = []
for i in range(len(pcs)-1):
    deltas.append(int(pcs[i+1] - pcs[i]))   # signed delta

# streaming windows
last_k = deque(maxlen=K)
win = deque(maxlen=WIN)
ngram_counts = Counter()
rows = []

# n-gram buffer
ngram_buf = deque(maxlen=NGRAM)

for t, d in enumerate(deltas):
    last_k.append(d)
    win.append(d)
    ngram_buf.append(d)
    if len(ngram_buf) == NGRAM:
        ngram_counts[tuple(ngram_buf)] += 1

    # only emit when we have at least K history
    if len(last_k) < K:
        continue

    # features
    feat = {}
    # raw last-k (most recent last, but we can store as delta_0=most recent)
    lk = list(last_k)
    for i in range(K):
        feat[f'delta_{i}'] = lk[-K + i] if len(lk) == K else (lk[i] if i < len(lk) else 0)

    # rolling stats
    feat['win_mean'] = float(np.mean(win)) if win else 0.0
    feat['win_std'] = float(np.std(win)) if win else 0.0
    feat['win_entropy'] = entropy_of_list(list(win))
    # run-length of last repeated delta
    runlen = 1
    it = reversed(list(last_k))
    prev = None
    for val in it:
        if prev is None:
            prev = val
            continue
        if val == prev:
            runlen += 1
        else:
            break
    feat['runlen'] = runlen
    # autocorr lag-1 in window
    arr = np.array(win, dtype=float)
    if len(arr) > 1:
        arr = arr - arr.mean()
        autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0,1])
    else:
        autocorr = 0.0
    feat['autocorr_lag1'] = 0.0 if math.isnan(autocorr) else autocorr
    # FFT low-frequency energy fraction
    feat['fft_lowfrac'] = fft_lowfreq_fraction(list(win), keep_frac=0.2)

    # n-gram top-3 counts (from the entire history so far)
    top3 = ngram_counts.most_common(3)
    for i in range(3):
        feat[f'ngram_top{i+1}_cnt'] = top3[i][1] if i < len(top3) else 0

    # label: the target delta (next delta) is deltas[t+1] if available; else empty
    target = deltas[t+1] if (t+1) < len(deltas) else None
    feat['target_delta'] = int(target) if target is not None else ''

    # optional: include current pc and next pc (for debugging)
    # we use pcs index t+1 as the 'current instruction IP' (aligning deltas -> PCs)
    feat['pc'] = pcs[t+1]
    feat['pc_next'] = pcs[t+2] if (t+2) < len(pcs) else 0

    rows.append(feat)

# write CSV
fieldnames = ['pc','pc_next'] + [f'delta_{i}' for i in range(K)] + ['win_mean','win_std','win_entropy','runlen','autocorr_lag1','fft_lowfrac',
            'ngram_top1_cnt','ngram_top2_cnt','ngram_top3_cnt','target_delta']

with open(OUT_CSV, 'w', newline='') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        # ensure keys exist
        out = {k: r.get(k, 0) for k in fieldnames}
        writer.writerow(out)

print(f'Wrote {len(rows)} rows to {OUT_CSV}')