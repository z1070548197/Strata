#!/usr/bin/env python3
"""Compare the solo and TP prefill hash logs (MOE_HASH / LAYER_HASH / GDN_HASH)."""
import re
import sys
from collections import defaultdict

MOE = re.compile(r"MOE_HASH rank=(-?\d+) p0=(\d+) stage=(\S+) cols=(\d+) last=([0-9a-f]+)")
LYR = re.compile(r"LAYER_HASH rank=(-?\d+) p0=(\d+) layer=(\d+) half=(\d+) last=([0-9a-f]+)")
GDN = re.compile(r"GDN_HASH rank=(-?\d+) p0=(\d+) stage=(\S+) bytes=(\d+) last=([0-9a-f]+)")


def load(path):
    moe, lyr, gdn = {}, {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = MOE.search(line)
            if m:
                rank, p0, stage, cols, h = m.groups()
                moe[(int(p0), stage, int(rank))] = h
                continue
            m = LYR.search(line)
            if m:
                rank, p0, layer, half, h = m.groups()
                lyr[(int(p0), int(layer), int(half), int(rank))] = h
                continue
            m = GDN.search(line)
            if m:
                rank, p0, stage, cols, h = m.groups()
                gdn[(int(p0), stage, int(rank))] = (h, int(cols))
    return moe, lyr, gdn


def main(solo_path, tp_path):
    s_moe, s_lyr, s_gdn = load(solo_path)
    t_moe, t_lyr, t_gdn = load(tp_path)
    print(f"solo: moe={len(s_moe)} layer={len(s_lyr)} gdn={len(s_gdn)}")
    print(f"tp  : moe={len(t_moe)} layer={len(t_lyr)} gdn={len(t_gdn)}")

    # --- MoE: solo's canonical rank halves correspond to TP's rank-local stages ---
    remap = {"gu_gate_r0": ("gu_gate_local", 0), "gu_gate_r1": ("gu_gate_local", 1),
             "gu_up_r0": ("gu_up_local", 0), "gu_up_r1": ("gu_up_local", 1),
             "h_r0": ("h_local", 0), "h_r1": ("h_local", 1)}
    solo_n = defaultdict(dict)
    for (p0, stage, rank), h in s_moe.items():
        if stage in remap:
            tstage, trank = remap[stage]
            solo_n[(p0, tstage, trank)] = h
        else:
            solo_n[(p0, stage, rank if rank >= 0 else 0)] = h
    keys = sorted(set(solo_n) & set(t_moe))
    bad = [(k, solo_n[k], t_moe[k]) for k in keys if solo_n[k] != t_moe[k]]
    only_s = sorted(set(solo_n) - set(t_moe))
    only_t = sorted(set(t_moe) - set(solo_n))
    print(f"MOE compared={len(keys)} mismatches={len(bad)} solo_only={len(only_s)} tp_only={len(only_t)}")
    for k, a, b in bad[:12]:
        print(f"  MOE MISMATCH {k}: solo={a} tp={b}")
    if only_s[:4]:
        print(f"  solo-only sample: {only_s[:4]}")
    if only_t[:4]:
        print(f"  tp-only sample: {only_t[:4]}")

    # --- GDN: solo runs full width; under TP most buffers are rank-local half width, so only stages whose
    # byte length agrees on both sides are comparable (y_full is the assembled full result). ---
    for rank in (0, 1):
        t = {(p0, stage): v for (p0, stage, r), v in t_gdn.items() if r == rank}
        s = {(p0, stage): v for (p0, stage, r), v in s_gdn.items() if r == -1}
        keys = sorted(k for k in set(s) & set(t) if s[k][1] == t[k][1])
        skipped = sorted({k[1] for k in set(s) & set(t) if s[k][1] != t[k][1]})
        bad = [(k, s[k][0], t[k][0]) for k in keys if s[k][0] != t[k][0]]
        print(f"GDN vs rank{rank}: compared={len(keys)} mismatches={len(bad)} "
              f"(rank-local width, not comparable: {','.join(skipped) or '-'})")
        for k, a, b in bad[:8]:
            print(f"  GDN MISMATCH {k}: solo={a} tp={b}")

    # --- LAYER: solo vs each TP rank ---
    for rank in (0, 1):
        t = {(p0, l, hf): v for (p0, l, hf, r), v in t_lyr.items() if r == rank}
        s = {(p0, l, hf): v for (p0, l, hf, r), v in s_lyr.items() if r == -1}
        keys = sorted(set(s) & set(t))
        bad = [(k, s[k], t[k]) for k in keys if s[k] != t[k]]
        print(f"LAYER vs rank{rank}: compared={len(keys)} mismatches={len(bad)} solo_only={len(set(s)-set(t))} tp_only={len(set(t)-set(s))}")
        for k, a, b in sorted(bad)[:15]:
            print(f"  LYR MISMATCH p0={k[0]} layer={k[1]} half={k[2]}: solo={a} tp={b}")
    # TP rank0 vs rank1 (the broadcast residual must be identical)
    r0 = {(p0, l, hf): v for (p0, l, hf, r), v in t_lyr.items() if r == 0}
    r1 = {(p0, l, hf): v for (p0, l, hf, r), v in t_lyr.items() if r == 1}
    keys = sorted(set(r0) & set(r1))
    bad = [k for k in keys if r0[k] != r1[k]]
    print(f"TP rank0 vs rank1: compared={len(keys)} mismatches={len(bad)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
