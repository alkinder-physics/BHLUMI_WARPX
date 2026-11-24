#!/usr/bin/env python3
"""
tier0_checks.py -- Tier 0 of the BHLUMI + WarpX compatibility test plan: checks that need no WarpX run.

    python3 tests/tier0_checks.py                                   # seed_2 sample, head-on benchmark input
    python3 tests/tier0_checks.py --compare $PSCRATCH/bhlumi/fcc_z/seed_3        # adds A8 (needs seed 3)
    python3 tests/tier0_checks.py --fcc-inputs ~/projects/FCC_pT_plots/wx_tier1_template.txt   # adds B6

Generator (A2-A8), conversion and frames (B1-B6), and the vertex sampler (C6, C7) on the head-on benchmark
input. Runs on a login node in about a minute; needs only numpy. Prints one PASS / FAIL / INFO line per test
and exits with status 1 if any test fails.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import wx_bhabha as wx  # noqa: E402  (pure numpy at import time; pywarpx is only loaded to run WarpX)

ALPHA = 1 / 137.035999
HBARC2_NB = 0.3893794e6        # (hbar c)^2 in nb GeV^2
M_E_GEV = 0.51099895e-3

results = []


def report(test, ok, msg):
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    results.append((test, tag))
    print(f"{tag:4s}  {test:4s}  {msg}")


def polar(p4, beam_sign):
    """Polar angle from the lepton's own beam direction: e- beam along -z (sign -1), e+ along +z (+1)."""
    p = np.linalg.norm(p4[:, :3], axis=1)
    return np.arccos(np.clip(beam_sign * p4[:, 2] / p, -1, 1))


def azimuth(p4):
    return np.arctan2(p4[:, 1], p4[:, 0])


# ----------------------------------------------------------------------------- A: generator
def check_generator(ev, run_dir, th1, th2, nmax_lhe):
    ele, pos = ev["ele"], ev["pos"]
    n = len(ele)
    sqrt_s = float(np.median(event_sums(ele.copy(), pos, ev["ph"], ev["ph_evt"])[:, 3]))   # 91.2 GeV
    s = sqrt_s ** 2
    xs, rel = float(ev["xsec_nb"]), float(ev["rel_err"])
    born = 16 * math.pi * ALPHA ** 2 / s * (1 / th1 ** 2 - 1 / th2 ** 2) * HBARC2_NB
    d = xs / born - 1
    report("A2", abs(d) < 0.05, f"sigma = {xs:.4g} +- {xs * rel:.2g} nb vs Born {born:.4g} nb "
                                f"({100 * d:+.2f} %; few % expected from radiative corrections)")

    te, tp = polar(ele, -1), polar(pos, +1)
    th = np.concatenate([te, tp])
    th_in = th[(th >= th1) & (th <= th2)]
    # maximum-likelihood exponent of dN/dtheta ~ theta^n truncated to [th1, th2]
    grid = np.linspace(-4, -2, 2001)
    lsum = np.log(th_in).sum()
    def norm(k):
        return np.log((th2 ** (k + 1) - th1 ** (k + 1)) / (k + 1))
    ll = np.array([k * lsum - len(th_in) * norm(k) for k in grid])
    k_hat = grid[np.argmax(ll)]
    curv = (ll[np.argmax(ll) + 1] - 2 * ll.max() + ll[np.argmax(ll) - 1]) / (grid[1] - grid[0]) ** 2
    k_err = 1 / math.sqrt(-curv) if curv < 0 else float("nan")
    edges = np.array([th1, 0.074, 0.108, th2])
    F = lambda x: (1 / th1 ** 2 - 1 / x ** 2) / (1 / th1 ** 2 - 1 / th2 ** 2)
    pred = np.diff([F(e) for e in edges])
    obs = np.histogram(th_in, edges)[0] / len(th_in)
    report("A3", abs(k_hat + 3) < max(0.05, 3 * k_err),
           f"dN/dtheta ~ theta^n, n = {k_hat:.3f} +- {k_err:.3f} (Born -3); fractions "
           f"{' / '.join(f'{o:.3f}' for o in obs)} vs {' / '.join(f'{p:.3f}' for p in pred)}; "
           f"{100 * (1 - len(th_in) / len(th)):.1f} % of leptons outside the window")

    nb = 36
    h = np.histogram(azimuth(ele), nb, (-math.pi, math.pi))[0]
    chi2 = ((h - n / nb) ** 2 / (n / nb)).sum()
    report("A4", chi2 / (nb - 1) < 2.0, f"e- azimuth: chi2/ndf = {chi2 / (nb - 1):.2f} for a flat distribution")

    emax = np.zeros(n)
    if len(ev["ph"]):
        np.maximum.at(emax, ev["ph_evt"], ev["ph"][:, 3])
    soft = emax < 1.0
    dphi = np.abs(np.angle(np.exp(1j * (azimuth(ele[soft]) - azimuth(pos[soft])))))
    acop = math.pi - dphi
    dth = np.abs(te[soft] - tp[soft]) / te[soft]
    q_acop, q_dth = np.quantile(acop, 0.99), np.quantile(dth, 0.99)
    report("A5", q_acop < 0.05 and q_dth < 0.05,
           f"{soft.mean() * 100:.1f} % of events have no photon above 1 GeV; for those, 99 % have "
           f"acoplanarity < {q_acop * 1e3:.2f} mrad and |theta(e-) - theta(e+)|/theta < {q_dth:.4f}")

    El = np.concatenate([ele[:, 3], pos[:, 3]])
    ebeam = sqrt_s / 2
    peak = np.mean(np.abs(El / ebeam - 1) < 0.01)
    above = np.mean(El > ebeam * (1 + 1e-6))
    report("A6", above == 0 and peak > 0.5,
           f"{100 * peak:.1f} % of leptons within 1 % of {ebeam:.4g} GeV; {100 * above:.3g} % above it; "
           f"median E' = {np.median(El):.4g} GeV, 1 % quantile {np.quantile(El, 0.01):.3g} GeV")

    lhe = os.path.join(run_dir, "out.lhe")
    if os.path.isfile(lhe):
        w, header_next = [], False
        with open(lhe) as f:
            for line in f:
                s_ = line.strip()
                if header_next:
                    w.append(float(s_.split()[2]))
                    header_next = False
                    if len(w) >= nmax_lhe:
                        break
                elif s_ == "<event>":
                    header_next = True
        w = np.array(w)
        report("A7", np.ptp(w) <= 1e-12 * abs(w).max(),
               f"{len(w)} event weights read: min {w.min():.6g}, max {w.max():.6g}")
    else:
        report("A7", None, f"skipped: no {lhe}")
    return sqrt_s


def compare_seeds(ev, ev2):
    xs1, r1 = float(ev["xsec_nb"]), float(ev["rel_err"])
    xs2, r2 = float(ev2["xsec_nb"]), float(ev2["rel_err"])
    pull = (xs1 - xs2) / math.hypot(xs1 * r1, xs2 * r2)
    edges = np.linspace(0.040, 0.120, 21)
    h1 = np.histogram(polar(ev["ele"], -1), edges)[0].astype(float)
    h2 = np.histogram(polar(ev2["ele"], -1), edges)[0].astype(float)
    n1, n2 = h1.sum(), h2.sum()
    ok = (h1 + h2) > 0
    chi2 = (((h1 / n1 - h2 / n2) ** 2) / (h1 / n1 ** 2 + h2 / n2 ** 2 + 1e-300))[ok].sum()
    ndf = ok.sum() - 1
    report("A8", abs(pull) < 3 and chi2 / ndf < 2,
           f"cross sections {xs1:.4g} vs {xs2:.4g} nb (pull {pull:+.2f}); theta shapes chi2/ndf = {chi2 / ndf:.2f}")


# ----------------------------------------------------------------------------- B: conversion and frames
def event_sums(ele, pos, ph, ph_evt):
    tot = ele + pos
    if len(ph):
        np.add.at(tot, ph_evt, ph)
    return tot


def check_conversion(ev, evfile, sqrt_s):
    ele, pos, ph, ph_evt = ev["ele"], ev["pos"], ev["ph"], ev["ph_evt"]
    tot = event_sums(ele.copy(), pos, ph, ph_evt)
    dp = np.linalg.norm(tot[:, :3], axis=1).max() / sqrt_s
    dE = np.abs(tot[:, 3] - np.median(tot[:, 3])).max() / sqrt_s
    report("B1", dp < 1e-5 and dE < 1e-5,
           f"per-event sum: max |sum p| / sqrt(s) = {dp:.2g}, max |sum E - {np.median(tot[:, 3]):.6g} GeV| / sqrt(s) = {dE:.2g}")

    def offshell(p4):
        return np.abs(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1)) / p4[:, 3] ** 2
    mlep = max(offshell(ele).max(), offshell(pos).max())
    mph = offshell(ph).max() if len(ph) else 0.0
    report("B2", mlep < 1e-6 and mph < 1e-6,
           f"max |E^2 - p^2| / E^2: leptons {mlep:.2g}, photons {mph:.2g} (m_e^2/E^2 = {(M_E_GEV / 45.6) ** 2:.2g})")

    ux, uy, uz = wx.u_from_p(ele[:1000])
    u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2) / wx.CLIGHT
    p = np.linalg.norm(ele[:1000, :3], axis=1)
    r = np.abs(u / (p / M_E_GEV) - 1).max()
    cosang = (ux * ele[:1000, 0] + uy * ele[:1000, 1] + uz * ele[:1000, 2]) / (u * wx.CLIGHT * p)
    report("B3", r < 1e-6 and np.all(cosang > 1 - 1e-12),
           f"|u|/c vs p/(m_e c): max relative difference {r:.2g}; e.g. {u[0]:.6g} for p = {p[0]:.6g} GeV; "
           f"direction preserved")

    rng = np.random.default_rng(1)
    n = min(20000, len(ele))
    e2, p2, ph2, phe2, w, _ = wx.prepare_events(evfile, n, rng, 1.0)
    report("B4", np.all(e2[:, 2] > 0) and np.all(p2[:, 2] < 0),
           f"after mirroring: {np.mean(e2[:, 2] > 0) * 100:.2f} % of e- move along +z (with beam1), "
           f"{np.mean(p2[:, 2] < 0) * 100:.2f} % of e+ along -z")
    tot2 = event_sums(e2.copy(), p2, ph2, phe2)
    dp2 = np.linalg.norm(tot2[:, :3], axis=1).max() / sqrt_s
    report("B5", dp2 < 1e-5, f"injected events in the WarpX frame: max |sum p| / sqrt(s) = {dp2:.2g} "
                             f"(head-on frame = CM frame, no boost needed)")


def check_crossing(path):
    pairs = wx.read_inputs(path)
    c = wx.eval_constants(pairs)
    phi = c["rotation_angle"]
    out = []
    for sgn in (+1, -1):
        cc = dict(c, rotation_angle=sgn * phi)
        b = wx.Bunches(cc)
        t, x, y, z, L, _ = b.sample(20000, np.random.default_rng(7))
        out.append((x, L, b.geometric_lumi()))
    (xp, Lp, Gp), (xm, Lm, _) = out
    q = np.linspace(0.05, 0.95, 19)
    asym = np.abs(np.quantile(xp, q) + np.quantile(xm, 1 - q)).max() / np.std(xp)
    dL = abs(Lp / Lm - 1)
    report("B6", asym < 0.1 and dL < 0.02,
           f"half crossing {phi:.4g} rad: x quantiles of +phi and -phi vertices mirror to {asym:.3f} sigma; "
           f"L_BX {Lp:.4g} vs {Lm:.4g} m^-2; geometric (Piwinski, no hourglass) {Gp:.4g}")


# ----------------------------------------------------------------------------- C6, C7: vertex sampler
def check_sampler(path, n):
    c = wx.eval_constants(wx.read_inputs(path))
    b = wx.Bunches(c)
    t, x, y, z, L, ess = b.sample(n, np.random.default_rng(3))
    sx, sy, sz = b.sx / math.sqrt(2), b.sy / math.sqrt(2), b.sz / math.sqrt(2)
    st = b.sz / (math.sqrt(2) * wx.CLIGHT)
    tol = 4 / math.sqrt(2 * n)                       # ~4 sigma on a sample standard deviation
    meas = [np.std(x) / sx, np.std(y) / sy, np.std(z) / sz, np.std(t) / st]
    ok = all(abs(m - 1) < tol for m in meas) and abs(np.mean(t) - b.tc) < 4 * st / math.sqrt(n)
    dt = c.get("dt")
    steps = f"; mean vertex at step {np.mean(t) / dt:.1f}, sigma_t = {np.std(t) / dt:.1f} steps" if dt else ""
    report("C6", ok, "vertex spread / (beam sigma / sqrt 2): " + ", ".join(
        f"{k} {m:.4f}" for k, m in zip(("x", "y", "z", "t"), meas)) + steps)
    L0 = b.N1 * b.N2 / (4 * math.pi * b.sx * b.sy)
    report("C7", abs(L / L0 - 1) < 0.01,
           f"sampler L_BX = {L:.4g} m^-2 vs N^2/(4 pi sigma_x sigma_y) = {L0:.4g} ({100 * (L / L0 - 1):+.2f} %); "
           f"effective samples {ess:.3g}")


def main():
    ps = os.environ.get("PSCRATCH", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=os.path.join(ps, "bhlumi/fcc_z/seed_2"),
                    help="BHLUMI run folder with events.npz (and out.lhe)")
    ap.add_argument("--compare", help="a second BHLUMI run folder (another seed) for A8")
    ap.add_argument("--inputs", default=os.path.join(ROOT, "benchmarks/headon_round/inputs_headon_round.txt"),
                    help="head-on benchmark input for C6, C7")
    ap.add_argument("--fcc-inputs", help="crossing-angle input (FCC template) for B6")
    ap.add_argument("--theta", nargs=2, type=float, default=(0.040, 0.120), help="card window [rad]")
    ap.add_argument("--n-lhe", type=int, default=100000, help="events whose weight is read for A7")
    a = ap.parse_args()

    evfile = os.path.join(a.sample, "events.npz")
    if not os.path.isfile(evfile):
        sys.exit(f"no {evfile}: run lhe_to_npz.py on the sample first")
    ev = np.load(evfile)
    print(f"sample {a.sample}: {len(ev['ele'])} events, {len(ev['ph'])} photons\n")
    sqrt_s = check_generator(ev, a.sample, *a.theta, a.n_lhe)
    if a.compare:
        compare_seeds(ev, np.load(os.path.join(a.compare, "events.npz")))
    else:
        report("A8", None, "skipped: pass --compare <second seed folder>")
    check_conversion(ev, evfile, sqrt_s)
    if a.fcc_inputs:
        check_crossing(a.fcc_inputs)
    else:
        report("B6", None, "skipped: pass --fcc-inputs <crossing-angle input>")
    check_sampler(a.inputs, 20000)

    fails = [t for t, s in results if s == "FAIL"]
    print(f"\n{sum(s == 'PASS' for _, s in results)} passed, {len(fails)} failed"
          + (f": {' '.join(fails)}" if fails else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
