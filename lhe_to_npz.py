#!/usr/bin/env python3
"""
lhe_to_npz.py -- convert a BHLUMI Les Houches file (out.lhe) to a compact NumPy file for WarpX.

    python3 lhe_to_npz.py $PSCRATCH/bhlumi/fcc_z/seed_2            # reads out.lhe + xsec.txt there
    python3 lhe_to_npz.py <run_dir> -o events.npz

Writes <run_dir>/events.npz with
    ele, pos   (N, 4)  final-state e- and e+:  px, py, pz, E   [GeV], BHLUMI frame (e- beam along -z)
    ph_evt     (M,)    event index of each final-state photon
    ph         (M, 4)  photons: px, py, pz, E [GeV]
    xsec_nb, rel_err, nevgen   from xsec.txt (each event represents xsec_nb / nevgen nb)
Only final-state particles (LHE status 1) are kept; every event must hold exactly one e- and one e+.
"""
import argparse
import os
import sys

import numpy as np


def read_xsec(path):
    vals = {}
    for line in open(path):
        k, v = line.split()
        vals[k] = float(v)
    return vals["XSEC_NB"], vals["REL_ERR"], int(vals["NEVGEN"])


def convert(lhe, out, xsec_file):
    ele, pos, ph, ph_evt = [], [], [], []
    n_ev, bad = 0, 0
    inside, cur_e, cur_p, cur_g = False, [], [], []
    with open(lhe) as f:
        for line in f:
            s = line.strip()
            if s == "<event>":
                inside, first, cur_e, cur_p, cur_g = True, True, [], [], []
                continue
            if s == "</event>":
                inside = False
                if len(cur_e) == 1 and len(cur_p) == 1:
                    ele.append(cur_e[0]); pos.append(cur_p[0])
                    for g in cur_g:
                        ph.append(g); ph_evt.append(n_ev)
                    n_ev += 1
                else:
                    bad += 1
                continue
            if not inside:
                continue
            if first:                      # event header line
                first = False
                continue
            c = s.split()
            if len(c) < 10 or c[1] != "1":  # keep final state only (status 1)
                continue
            pid = int(c[0])
            p4 = [float(c[6]), float(c[7]), float(c[8]), float(c[9])]
            if pid == 11:
                cur_e.append(p4)
            elif pid == -11:
                cur_p.append(p4)
            elif pid == 22:
                cur_g.append(p4)
    xs, rel, nev = read_xsec(xsec_file)
    np.savez(out, ele=np.array(ele, dtype=np.float64), pos=np.array(pos, dtype=np.float64),
             ph=np.array(ph, dtype=np.float64).reshape(-1, 4), ph_evt=np.array(ph_evt, dtype=np.int64),
             xsec_nb=xs, rel_err=rel, nevgen=nev)
    return n_ev, bad, len(ph), xs, nev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="BHLUMI run folder holding out.lhe and xsec.txt")
    ap.add_argument("-o", "--out", help="output .npz (default: <run_dir>/events.npz)")
    a = ap.parse_args()
    lhe, xsf = os.path.join(a.run_dir, "out.lhe"), os.path.join(a.run_dir, "xsec.txt")
    for p in (lhe, xsf):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: {p} not found")
    out = a.out or os.path.join(a.run_dir, "events.npz")
    n, bad, nph, xs, nev = convert(lhe, out, xsf)
    print(f"{out}: {n} events ({bad} skipped), {nph} photons; xsec = {xs:.6g} nb from {nev} generated")
    if n != nev:
        print(f"WARNING: {n} events kept but NEVGEN = {nev}; check out.lhe")
