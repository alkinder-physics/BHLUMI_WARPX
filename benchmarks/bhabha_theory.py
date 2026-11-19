"""
bhabha_theory.py -- analytic prediction for the head-on round-beam Bhabha benchmark, read from a WarpX
input file so the theory and the simulation always use the same numbers.

Physics: a Bhabha lepton (e- from beam1, moving along +z) leaves its vertex at polar angle theta and crosses
the rest of the opposite (e+) bunch, which pulls it toward the axis. For two round Gaussian bunches
(sigma_x = sigma_y = sigma, length sigma_z, N particles each) colliding head-on, rigid (disruption << 1),
without hourglass or beamstrahlung, the kick averaged over the luminous region is

    <dtheta>(theta) = - N r_e / (sqrt(pi) gamma' theta sigma_z) * ln(1 + theta^2 sigma_z^2 / (3 sigma^2))

with gamma' the Lorentz factor of the outgoing lepton (derivation in notebooks/01_headon_round_benchmark).
The same formula holds for the e+ with the roles of the beams swapped.

Everything here is computed from the WarpX my_constants block (names as in the FCC box-scan input):
    energy, npart, sigmax, sigmay, sigmaz, rotation_angle, betax, betay, muz,
    Lx, Ly, Lz, nx, ny, nz, dt, and stop_time (or T)
"""
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from wx_bhabha import CLIGHT, M_E, Q_E, eval_constants, read_inputs  # noqa: E402

R_E = 2.8179403262e-15                 # classical electron radius [m]
ME_GEV = M_E * CLIGHT ** 2 / (1e9 * Q_E)


# ----------------------------------------------------------------------------- parameters
class Params:
    """Benchmark parameters evaluated from a WarpX input file (plus optional --set style overrides)."""

    def __init__(self, path, overrides=()):
        self.path = path
        self.pairs = read_inputs(path) + list(overrides)
        self.raw = dict(self.pairs)                     # last definition wins, as in WarpX
        c = self.c = eval_constants(self.pairs)
        self.E_J = c["energy"]
        self.E_GeV = self.E_J / (1e9 * Q_E)
        self.gamma = self.E_J / (M_E * CLIGHT ** 2)
        self.N = c["npart"]
        self.sx, self.sy, self.sz = c["sigmax"], c["sigmay"], c["sigmaz"]
        self.sigma = 0.5 * (self.sx + self.sy)
        self.phi = c.get("rotation_angle", 0.0)
        self.betax, self.betay = c.get("betax", math.inf), c.get("betay", math.inf)
        self.muz = c["muz"]
        self.Lx, self.Ly, self.Lz = c["Lx"], c["Ly"], c["Lz"]
        self.nx, self.ny, self.nz = int(c["nx"]), int(c["ny"]), int(c["nz"])
        self.dx, self.dy, self.dz = self.Lx / self.nx, self.Ly / self.ny, self.Lz / self.nz
        self.dt = c["dt"]
        self.stop_time = self._number(self.raw.get("stop_time", "T"))
        self.zcut = self._number(self.raw.get("beam1.z_cut", "inf"))

    def _number(self, expr):
        env = {**self.c, "inf": math.inf, "clight": CLIGHT, "sqrt": math.sqrt, "floor": math.floor}
        return float(eval(expr.replace("^", "**"), {"__builtins__": {}}, env))

    def table(self):
        rows = [("source file", self.path, ""),
                ("beam energy", f"{self.E_GeV:.6g}", "GeV"), ("gamma", f"{self.gamma:.6g}", ""),
                ("N per bunch", f"{self.N:.4g}", ""),
                ("sigma_x, sigma_y", f"{self.sx * 1e6:.4g}, {self.sy * 1e6:.4g}", "um"),
                ("sigma_z", f"{self.sz * 1e3:.4g}", "mm"),
                ("half crossing angle", f"{self.phi:.4g}", "rad"),
                ("box Lx, Ly", f"{self.Lx * 1e6:.4g}, {self.Ly * 1e6:.4g}", "um"),
                ("box Lz", f"{self.Lz * 1e3:.4g}", "mm"),
                ("cells nx, ny, nz", f"{self.nx}, {self.ny}, {self.nz}", ""),
                ("dx, dz, c dt", f"{self.dx * 1e6:.3g} um, {self.dz * 1e6:.3g} um, "
                                 f"{CLIGHT * self.dt * 1e6:.3g} um", ""),
                ("stop time x c", f"{self.stop_time * CLIGHT * 1e3:.4g}", "mm")]
        return rows


# ----------------------------------------------------------------------------- theory
def disruption(p):
    return 2 * p.N * R_E * p.sz / (p.gamma * p.sigma ** 2)


def luminosity_per_crossing(p):
    """Luminosity per crossing [m^-2] for two equal Gaussian beams, head-on, no hourglass: N^2/(4 pi sx sy)."""
    return p.N ** 2 / (4 * math.pi * p.sx * p.sy)


def kick_closed_form(theta, p, E_lepton_GeV=None):
    """Mean kick [rad] at polar angle theta [rad] (infinite box, untruncated Gaussians)."""
    theta = np.asarray(theta, dtype=float)
    g = p.gamma if E_lepton_GeV is None else np.asarray(E_lepton_GeV) / ME_GEV
    x = theta ** 2 * p.sz ** 2 / (3 * p.sigma ** 2)
    return -p.N * R_E / (math.sqrt(math.pi) * g * theta * p.sz) * np.log1p(x)


def cutoff_length(theta, p):
    """Distance along the opposite bunch over which the lepton leaves the core: 2 sqrt(3) sigma / theta."""
    return 2 * math.sqrt(3) * p.sigma / np.asarray(theta, dtype=float)


def kick_mc(theta, p, n_events=20000, n_u=4000, seed=1, finite_box=True, chunk=500):
    """Direct numerical average of the kick over vertices and trajectories.

    Same physics as the closed form, but it can include what the simulation has and the formula does not:
    the transverse and longitudinal box walls (the lepton is absorbed there and its kick stops), the end of
    the run, the bunches' z cut, and sigma_x != sigma_y (field of an elliptical bunch, evaluated
    numerically only when they differ). Returns (mean, standard error, rms) in rad.
    """
    rng = np.random.default_rng(seed)
    zc = p.zcut if np.isfinite(p.zcut) else 8.0
    norm = 0.5 * (math.erf(zc / math.sqrt(2)) - math.erf(-zc / math.sqrt(2)))
    phi = rng.uniform(0, 2 * np.pi, n_events)
    nx_, ny_ = np.cos(phi), np.sin(phi)
    x0 = rng.normal(0, p.sx / math.sqrt(2), n_events)
    y0 = rng.normal(0, p.sy / math.sqrt(2), n_events)
    s1 = _trunc_normal(rng, p.sz, zc, n_events)          # colliding e- inside beam1
    z0 = _trunc_normal(rng, p.sz, zc, n_events)          # colliding e+ inside beam2 (zeta0)
    t_v = p.muz / CLIGHT + (z0 - s1) / (2 * CLIGHT)
    z_v = 0.5 * (s1 + z0)
    tot = np.zeros(n_events)
    round_ = abs(p.sx - p.sy) < 1e-9 * p.sx
    for k in range(0, n_events, chunk):
        sl = slice(k, k + chunk)
        umax = (zc * p.sz - z0[sl])[:, None]             # rest of the opposite bunch
        u = np.linspace(0, 1, n_u)[None, :] * umax
        du = umax[:, 0] / (n_u - 1)
        rx = x0[sl, None] + nx_[sl, None] * theta * u / 2
        ry = y0[sl, None] + ny_[sl, None] * theta * u / 2
        lam = np.exp(-0.5 * ((z0[sl, None] + u) / p.sz) ** 2) / (math.sqrt(2 * math.pi) * p.sz * norm)
        if round_:
            r2 = rx ** 2 + ry ** 2
            f = np.where(r2 > 0, (1 - np.exp(-r2 / (2 * p.sx ** 2))) / np.maximum(r2, 1e-300), 1 / (2 * p.sx ** 2))
            ex, ey = rx * f, ry * f                       # 2 pi eps0 E / lambda
        else:
            ex, ey = _bassetti_erskine_like(rx, ry, p.sx, p.sy)
        integrand = (ex * nx_[sl, None] + ey * ny_[sl, None]) * lam
        if finite_box:
            t = t_v[sl, None] + u / (2 * CLIGHT)
            z = z_v[sl, None] + CLIGHT * np.cos(theta) * u / (2 * CLIGHT)
            inside = ((np.abs(rx) < p.Lx / 2) & (np.abs(ry) < p.Ly / 2) & (z < p.Lz / 2) & (t <= p.stop_time))
            inside = np.cumprod(inside, axis=1).astype(bool)    # absorbed at the first wall it meets
            integrand = integrand * inside
        tot[sl] = np.trapezoid(integrand, axis=1) * du
    dth = -(2 * p.N * R_E / p.gamma) * tot
    return dth.mean(), dth.std() / math.sqrt(n_events), dth.std()


def _trunc_normal(rng, s, cut, n):
    out = rng.normal(0, s, n)
    bad = np.abs(out) > cut * s
    while bad.any():
        out[bad] = rng.normal(0, s, bad.sum())
        bad = np.abs(out) > cut * s
    return out


def _bassetti_erskine_like(x, y, sx, sy, n_t=64):
    """Field of an elliptical Gaussian line charge (2 pi eps0 E / lambda), by the integral representation
    E ~ r * int_0^inf exp(-x^2/(2sx^2+q) - y^2/(2sy^2+q)) / ((2sx^2+q)^(3/2)(2sy^2+q)^(1/2)) dq."""
    # substitute q = s_max^2 * (1/w - 1)... simple Gauss-Legendre in w in (0,1]
    w, wt = np.polynomial.legendre.leggauss(n_t)
    w = 0.5 * (w + 1); wt = 0.5 * wt
    s2 = 2 * max(sx, sy) ** 2
    ex = np.zeros_like(x); ey = np.zeros_like(y)
    for wi, wti in zip(w, wt):
        q = s2 * (1 / wi - 1); dq = s2 / wi ** 2
        ax, ay = 2 * sx ** 2 + q, 2 * sy ** 2 + q
        e = np.exp(-x ** 2 / ax - y ** 2 / ay) / np.sqrt(ax * ay)
        ex += wti * dq * x / ax * e
        ey += wti * dq * y / ay * e
    return ex, ey


# ----------------------------------------------------------------------------- checks
def validity_checks(p, theta_min, theta_max, mc_events=4000):
    """List of (check, value, requirement, ok). The analytic formula assumes all of these."""
    out = []
    def add(name, val, req, ok):
        out.append((name, val, req, bool(ok)))
    rnd = abs(p.sx - p.sy) / p.sx
    add("round beams |sx-sy|/sx", f"{rnd:.2g}", "< 1e-6 (formula is for round beams)", rnd < 1e-6)
    add("head-on (half crossing)", f"{p.phi:.3g} rad", "= 0", p.phi == 0)
    D = disruption(p)
    add("disruption D = 2 N r_e sz / (gamma s^2)", f"{D:.3g}", "< 0.05 (rigid bunches)", D < 0.05)
    hg = min(p.betax, p.betay) / p.sz
    add("hourglass beta*/sigma_z (vertex sampler)", f"{hg:.3g}", ">= 100", hg >= 100)
    ux_th = p._number(p.raw.get("beam1.ux_th", "0")) + p._number(p.raw.get("beam1.uy_th", "0"))
    add("beam emittance (ux_th, uy_th)", f"{ux_th:.3g}", "= 0 (no hourglass in WarpX)",
        ux_th == 0 and "beam1.focal_distance" not in p.raw)
    qed = [k for k, v in p.raw.items() if "do_qed" in k and v.strip() not in ("0", "0.")]
    add("QED processes", ", ".join(qed) or "none", "none (no beamstrahlung)", not qed)
    bext = [k for k in p.raw if "B_ext" in k or "E_ext" in k or "external_particle_function" in k]
    add("external fields", ", ".join(bext) or "none", "none", not bext)
    add("transverse resolution sigma/dx", f"{p.sigma / max(p.dx, p.dy):.3g}", ">= 5",
        p.sigma / max(p.dx, p.dy) >= 5)
    lc = cutoff_length(theta_max, p)
    step = max(p.dz, CLIGHT * p.dt)
    add(f"longitudinal: cutoff length at {theta_max * 1e3:.0f} mrad / max(dz, c dt)",
        f"{lc / step:.3g}", ">= 10", lc / step >= 10)
    t_last = p.muz / CLIGHT + (p.zcut if np.isfinite(p.zcut) else 4) * p.sz / CLIGHT
    add("stop_time / last vertex time", f"{p.stop_time / t_last:.4g}", ">= 1", p.stop_time >= t_last * (1 - 1e-9))
    for th in (theta_min, theta_max):
        ideal = kick_mc(th, p, n_events=mc_events, finite_box=False, seed=2)[0]
        boxed = kick_mc(th, p, n_events=mc_events, finite_box=True, seed=2)[0]
        loss = 1 - boxed / ideal
        add(f"box truncation of the kick at {th * 1e3:.0f} mrad", f"{loss:.2g}", "< 1e-3", abs(loss) < 1e-3)
    return out


def print_table(rows, header=None):
    rows = [tuple(str(x) for x in r) for r in rows]
    if header:
        rows = [tuple(header)] + rows
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for j, r in enumerate(rows):
        print("  ".join(s.ljust(w[i]) for i, s in enumerate(r)))
        if header and j == 0:
            print("  ".join("-" * x for x in w))


# ----------------------------------------------------------------------------- run output
def load_run(run_dir, diags="diags"):
    """Per-lepton input and output angles of a benchmark run (needs `wx_bhabha.py --tag-weights`).

    Returns a dict of arrays for e- and e+ together: species (0 = e-, 1 = e+), theta0, phi0, E0 [GeV]
    (BHLUMI, after the z flip), theta_out, phi_out, where (0 = scraped at a wall, 1 = still in the box at
    the end), and ievt. theta is measured from each lepton's own forward axis (+z for e-, -z for e+).
    """
    inj = np.load(os.path.join(run_dir, "bhabha_injected.npz"))
    out = {k: [] for k in ("species", "ievt", "theta_out", "phi_out", "where")}
    for sp_i, sp in enumerate(("ele_bhl", "pos_bhl")):
        for where, w, px, py, pz in _read_species(os.path.join(run_dir, diags), sp):
            ievt = np.rint(w).astype(np.int64) - 1
            th, ph = _angles(px, py, pz, sp_i)
            out["species"].append(np.full(len(ievt), sp_i)); out["ievt"].append(ievt)
            out["theta_out"].append(th); out["phi_out"].append(ph); out["where"].append(np.full(len(ievt), where))
    out = {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}
    if out["ievt"].size == 0:
        raise RuntimeError(f"no ele_bhl/pos_bhl particles found under {run_dir}/{diags}")
    if out["ievt"].min() < 0 or out["ievt"].max() >= len(inj["ele"]):
        raise RuntimeError("weights are not event tags: run wx_bhabha.py with --tag-weights")
    p4 = np.where(out["species"][:, None] == 0, inj["ele"][out["ievt"]], inj["pos"][out["ievt"]])
    th0, ph0 = _angles(p4[:, 0], p4[:, 1], p4[:, 2], out["species"])
    out.update(theta0=th0, phi0=ph0, E0=p4[:, 3])
    out["dtheta"] = out["theta_out"] - out["theta0"]
    out["n_injected"] = len(inj["ele"])
    return out


def _angles(px, py, pz, species):
    sgn = np.where(np.asarray(species) == 0, 1.0, -1.0)
    pzf = sgn * pz
    pt = np.hypot(px, py)
    return np.arctan2(pt, pzf), np.arctan2(py, px)


def _read_species(diag_root, species):
    """Yield (where, weight, px, py, pz) for one species: every scraped particle (BoundaryScraping
    output) and, from the last iteration of each Full diagnostic, the particles still in the box."""
    import openpmd_api as io
    found = {}
    for f in glob.glob(os.path.join(diag_root, "**", "openpmd*"), recursive=True):
        d, name = os.path.split(f)
        ext = os.path.splitext(name)[1]
        if ext not in (".bp", ".bp4", ".bp5", ".h5"):
            continue
        # file-based encoding (openpmd_000123.bp, one per iteration) or one group-based file (openpmd.bp)
        found[d] = os.path.join(d, ("openpmd_%T" if name.startswith("openpmd_") else "openpmd") + ext)
    for d, pattern in sorted(found.items()):
        s = io.Series(pattern, io.Access.read_only)
        its = list(s.iterations)
        scraped = "particles_at_" in d or "scrap" in d.lower()
        use = its if scraped else its[-1:]
        for it in use:
            parts = s.iterations[it].particles
            if species not in parts:
                continue
            sp = parts[species]
            def rec(r, c):
                comp = sp[r][c] if c else sp[r][io.Record_Component.SCALAR]
                return comp.load_chunk(), comp.unit_SI
            got = [rec("weighting", None)] + [rec("momentum", c) for c in "xyz"]
            s.flush()
            w, px, py, pz = (np.asarray(a, dtype=float) * u for a, u in got)
            if w.size:
                yield (0 if scraped else 1), w, px, py, pz
        s.close()


def luminosity_from_reduced(run_dir, p, name="DiffLum_beam1_beam2"):
    """Total luminosity [m^-2] from WarpX's DifferentialLuminosity reduced diagnostic (last row), or None."""
    path = os.path.join(run_dir, "diags", "reducedfiles", f"{name}.txt")
    if not os.path.isfile(path):
        return None
    rows = np.atleast_2d(np.loadtxt(path))
    dE = p.c.get("bin_size_1d")
    if dE is None:
        return None
    return float(rows[-1, 2:].sum() * dE)       # columns: step, time, dL/dE per bin [m^-2 eV^-1]


def compare(run, p, bins):
    """Binned mean kick vs theta0 next to the closed form. Returns rows and arrays for plotting."""
    pred = kick_closed_form(run["theta0"], p, E_lepton_GeV=run["E0"])
    resid = run["dtheta"] - pred
    idx = np.digitize(run["theta0"], bins) - 1
    rows, arr = [], {k: [] for k in ("theta", "sim", "err", "theory", "n")}
    for i in range(len(bins) - 1):
        m = idx == i
        n = int(m.sum())
        if n < 10:
            continue
        sim, err = run["dtheta"][m].mean(), run["dtheta"][m].std() / math.sqrt(n)
        th = pred[m].mean()
        rows.append((f"{bins[i] * 1e3:.0f}-{bins[i + 1] * 1e3:.0f}", n, f"{sim * 1e6:.3f} +- {err * 1e6:.3f}",
                     f"{th * 1e6:.3f}", f"{(sim - th) / err:+.2f}"))
        for k, v in zip(arr, (run["theta0"][m].mean(), sim, err, th, n)):
            arr[k].append(v)
    total = (resid.mean(), resid.std() / math.sqrt(len(resid)))
    return rows, {k: np.array(v) for k, v in arr.items()}, total
