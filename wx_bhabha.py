#!/usr/bin/env python3
"""
wx_bhabha.py -- run a WarpX beam-beam simulation with BHLUMI Bhabha events injected at their vertices.

Steps (all ranks do the same, with the same random seed, so no communication is needed):
  1. read the WarpX input template and evaluate its my_constants (beam sizes, crossing angle, timing)
  2. write inputs_bhabha.txt: the template + two test-particle species, ele_bhl and pos_bhl
     (no charge deposition, saved in particles_in and at every wall like the pair species) + overrides
  3. sample one vertex (t, x, y, z) per injected event from the luminosity density of the two
     tilted, hourglass-focused Gaussian bunches, and estimate the luminosity per crossing L_BX
  4. take N BHLUMI events (lhe_to_npz.py output), mirror z (BHLUMI has the e- beam along -z, WarpX beam1
     e- along +z), give each the weight  sigma * L_BX / N
  5. run WarpX through pywarpx; before every step, add the events whose vertex time falls in that step
  Photons are not tracked (they are not deflected); they are written with their vertices to
  bhabha_injected.npz together with everything that was injected.

Usage
  python3 wx_bhabha.py --check  --inputs wx_tier1_template.txt --events events.npz      # no WarpX run
  srun ... python3 wx_bhabha.py --inputs wx_tier1_template.txt --events events.npz \\
           --n-inject 20000 --seed 1 --set my_constants.Bz=2 --set warpx.random_seed=1
  benchmarks: add --tag-weights so each scraped lepton can be matched to its injected event

Version 1 limits: events keep their nominal energies (no beamstrahlung rescaling yet); the bunch model for
the vertices is analytic (Gaussian + hourglass + tilt), not the simulated, disrupted bunches.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

CLIGHT = 299792458.0
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
GEV_TO_SI = 1e9 * Q_E / CLIGHT          # GeV/c -> kg m/s
NB_TO_M2 = 1e-37
NEW_SPECIES = {"ele_bhl": "electron", "pos_bhl": "positron"}
WALLS = ["xlo", "ylo", "zlo", "xhi", "yhi", "zhi"]


# ----------------------------------------------------------------------------- inputs
def read_inputs(path):
    """Ordered list of (key, value) from a WarpX inputs file (comments and blank lines dropped)."""
    out = []
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def eval_constants(pairs):
    """Evaluate my_constants.* the way WarpX's parser would (definition order does not matter)."""
    env = {"pi": math.pi, "clight": CLIGHT, "q_e": Q_E, "m_e": M_E, "m_p": 1.67262192369e-27,
           "epsilon0": 8.8541878128e-12, "mu0": 1.25663706212e-6, "kb": 1.380649e-23,
           "sqrt": math.sqrt, "cos": math.cos, "sin": math.sin, "tan": math.tan, "exp": math.exp,
           "log": math.log, "floor": math.floor, "ceil": math.ceil, "abs": abs, "min": min, "max": max,
           "atan": math.atan, "atan2": math.atan2, "pow": pow}
    todo = {k[len("my_constants."):]: v for k, v in pairs if k.startswith("my_constants.")}
    vals = {}
    while todo:
        progress = False
        for name, expr in list(todo.items()):
            try:
                vals[name] = float(eval(expr.replace("^", "**"), {"__builtins__": {}}, {**env, **vals}))
                del todo[name]; progress = True
            except NameError:
                continue
        if not progress:
            raise RuntimeError(f"cannot evaluate my_constants: {sorted(todo)}")
    return vals


def write_inputs(template_pairs, overrides, out_path):
    lines, have = [], {k for k, _ in template_pairs}
    for k, v in template_pairs:
        if k in ("particles.species_names", "particles_in.species"):
            v = v + " " + " ".join(NEW_SPECIES)
        lines.append(f"{k} = {v}")
    lines.append("# ---- BHLUMI Bhabha test particles (added by wx_bhabha.py)")
    for sp, kind in NEW_SPECIES.items():
        lines += [f"{sp}.species_type = {kind}", f"{sp}.injection_style = none", f"{sp}.do_not_deposit = 1"]
        lines += [f"{sp}.save_particles_at_{w} = 1" for w in WALLS]
    if "particles_in.species" not in have:
        print("WARNING: no particles_in.species in the template; ele_bhl/pos_bhl only saved at the walls")
    lines.append("# ---- command-line overrides")
    lines += [f"{k} = {v}" for k, v in overrides]
    tmp = f"{out_path}.tmp{os.getpid()}"             # write then rename: readers never see a partial file
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)


# ----------------------------------------------------------------------------- vertices
class Bunches:
    """Two Gaussian bunches in the WarpX head-on frame: beam1 from -muz along +z tilted by +phi,
    beam2 from +muz along -z tilted by -phi, both focused (hourglass) at z = 0."""

    def __init__(self, c):
        self.N1 = self.N2 = c["npart"]
        self.sx, self.sy, self.sz = c["sigmax"], c["sigmay"], c["sigmaz"]
        self.bx, self.by = c["betax"], c["betay"]
        self.phi, self.muz = c["rotation_angle"], c["muz"]
        self.tc = self.muz / CLIGHT                     # time at which the bunch centres cross

    def density(self, x, y, z, t, beam):
        if beam == 1:
            zc, a, N = -self.muz + CLIGHT * t, self.phi, self.N1
        else:
            zc, a, N = self.muz - CLIGHT * t, -self.phi, self.N2
        s = z - zc
        xb = x * np.cos(a) - s * np.sin(a)              # coordinates along the tilted bunch axes
        sb = x * np.sin(a) + s * np.cos(a)
        sx = self.sx * np.sqrt(1 + (z / self.bx) ** 2)  # hourglass
        sy = self.sy * np.sqrt(1 + (z / self.by) ** 2)
        return N / ((2 * np.pi) ** 1.5 * sx * sy * self.sz) * np.exp(
            -0.5 * (xb / sx) ** 2 - 0.5 * (y / sy) ** 2 - 0.5 * (sb / self.sz) ** 2)

    def sample(self, n, rng, n_prop=None):
        """Importance-sample n vertices from n1*n2; returns (t, x, y, z) arrays and L_BX [m^-2]."""
        n_prop = n_prop or max(50 * n, 2_000_000)
        szp = min(self.sz, 4 * self.sx / max(self.phi, 1e-12))   # luminous length ~ sigma_x / phi
        tp = self.tc + rng.normal(0, self.sz / CLIGHT, n_prop)
        zp = rng.normal(0, szp, n_prop)
        # In the head-on frame the overlap sweeps across x at |x0| = phi * c * |t - t_c| (the lab's
        # crossing angle seen from the boosted frame); the sign depends on the tilt convention, so the
        # proposal is a mixture centred at +x0 and -x0 (the density itself fixes the side).
        x0 = self.phi * CLIGHT * (tp - self.tc)
        wx = 1.5 * self.sx
        side = np.where(rng.random(n_prop) < 0.5, 1.0, -1.0)
        xp = side * x0 + rng.normal(0, wx, n_prop)
        syz = 2 * self.sy * np.sqrt(1 + (zp / self.by) ** 2)
        yp = rng.normal(0, syz, n_prop)
        g = lambda u, s: np.exp(-0.5 * (u / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        qx = 0.5 * (g(xp - x0, wx) + g(xp + x0, wx))
        q = g(tp - self.tc, self.sz / CLIGHT) * g(zp, szp) * qx * g(yp, syz)
        r = self.density(xp, yp, zp, tp, 1) * self.density(xp, yp, zp, tp, 2) / q
        L_bx = 2 * CLIGHT * r.mean()                    # L = 2c * integral n1 n2 dV dt
        ess = r.sum() ** 2 / (r ** 2).sum()
        idx = rng.choice(n_prop, size=n, replace=True, p=r / r.sum())
        return tp[idx], xp[idx], yp[idx], zp[idx], L_bx, ess

    def geometric_lumi(self):
        Phi = self.phi * self.sz / self.sx
        return self.N1 * self.N2 / (4 * np.pi * self.sx * self.sy * np.sqrt(1 + Phi ** 2))


# ----------------------------------------------------------------------------- events
def prepare_events(evfile, n, rng, L_bx):
    ev = np.load(evfile)
    n_avail = len(ev["ele"])
    if n > n_avail:
        sys.exit(f"ERROR: --n-inject {n} > {n_avail} events in {evfile}")
    pick = rng.choice(n_avail, size=n, replace=False)
    flip = np.array([1.0, 1.0, -1.0, 1.0])              # BHLUMI e- along -z -> WarpX beam1 e- along +z
    ele, pos = ev["ele"][pick] * flip, ev["pos"][pick] * flip
    sel = np.isin(ev["ph_evt"], pick)
    remap = {old: new for new, old in enumerate(pick)}
    ph_evt = np.array([remap[i] for i in ev["ph_evt"][sel]], dtype=np.int64)
    ph = ev["ph"][sel] * flip
    sigma_m2 = float(ev["xsec_nb"]) * NB_TO_M2
    w = sigma_m2 * L_bx / n                              # physical Bhabhas per injected event
    return ele, pos, ph, ph_evt, w, float(ev["xsec_nb"])


def u_from_p(p4):
    """GeV/c momentum -> WarpX u = gamma*v [m/s]."""
    return p4[:, 0] * GEV_TO_SI / M_E, p4[:, 1] * GEV_TO_SI / M_E, p4[:, 2] * GEV_TO_SI / M_E


# ----------------------------------------------------------------------------- WarpX
def warpx_api():
    from pywarpx import callbacks, particle_containers, warpx
    try:
        from pywarpx import libwarpx
    except ImportError:
        libwarpx = None
    return warpx, callbacks, particle_containers, libwarpx


def sim_time_dt(libwarpx):
    """Current time and dt from WarpX (raises if this pywarpx version offers neither accessor)."""
    so = libwarpx.libwarpx_so
    if hasattr(so, "get_instance"):
        inst = so.get_instance()
        return inst.gett_new(0), inst.getdt(0)
    return so.warpx_gett_new(0), so.warpx_getdt(0)


def _host_list(arrs):
    out = []
    for a_ in (arrs if isinstance(arrs, (list, tuple)) else [arrs]):
        out.append(a_.get() if hasattr(a_, "get") and not isinstance(a_, np.ndarray) else np.asarray(a_))
    return np.concatenate(out) if out else np.array([])


def beam1_tilt_sign(pcs):
    """Sign of d<x>/dz of beam1 as WarpX initialised it (summed over MPI ranks when mpi4py is there)."""
    pc = pcs.ParticleContainerWrapper("beam1")
    def get(name):
        f = getattr(pc, f"get_particle_{name}")
        try:
            return _host_list(f(copy_to_host=True))
        except TypeError:
            return _host_list(f())
    x, z = get("x"), get("z")
    sums = np.array([len(x), x.sum(), z.sum(), (x * z).sum(), (z * z).sum()], dtype=np.float64)
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
    except ImportError:
        pass
    n, sx, sz, sxz, szz = sums
    if n < 2:
        return 0.0
    cov = sxz / n - (sx / n) * (sz / n)
    return float(np.sign(cov))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="WarpX input template")
    ap.add_argument("--events", required=True, help="events.npz from lhe_to_npz.py")
    ap.add_argument("--n-inject", type=int, default=20000, help="BHLUMI events to inject")
    ap.add_argument("--seed", type=int, default=1, help="seed for vertex and event sampling")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="input override")
    ap.add_argument("--out", default=".", help="run directory")
    ap.add_argument("--check", action="store_true", help="check the setup and sampling, do not run WarpX")
    ap.add_argument("--tag-weights", action="store_true",
                    help="set each test particle's weight to (event index + 1) so the output can be matched "
                         "event by event (benchmarks); the physical weight stays in bhabha_injected.npz")
    a = ap.parse_args()

    overrides = [tuple(s.split("=", 1)) for s in a.set]
    pairs = read_inputs(a.inputs) + overrides
    c = eval_constants(pairs)
    b = Bunches(c)
    rng = np.random.default_rng(a.seed)
    t, x, y, z, L_bx, ess = b.sample(a.n_inject, rng)
    ele, pos, ph, ph_evt, w, xs = prepare_events(a.events, a.n_inject, rng, L_bx)

    os.makedirs(a.out, exist_ok=True)
    rank0 = os.environ.get("SLURM_PROCID", "0") == "0"
    if rank0:
        print(f"[wx_bhabha] N = {b.N1:.3g}, sigma_x,y,z = {b.sx:.3g} m, {b.sy:.3g} m, {b.sz:.3g} m, "
              f"beta_x,y = {b.bx:.3g}, {b.by:.3g} m, half crossing = {b.phi:.4g} rad")
        print(f"[wx_bhabha] L_BX = {L_bx:.4g} m^-2 (sampled, hourglass + tilt; geometric estimate "
              f"{b.geometric_lumi():.4g}); effective proposal samples {ess:.3g}")
        print(f"[wx_bhabha] sigma = {xs:.5g} nb -> {xs * NB_TO_M2 * L_bx:.4g} Bhabhas per crossing; "
              f"{a.n_inject} injected, weight {w:.4g} each")
        print(f"[wx_bhabha] vertex spread: c(t - t_c) {CLIGHT * np.std(t - b.tc) * 1e3:.3g} mm, "
              f"x {np.std(x) * 1e6:.3g} um, y {np.std(y) * 1e9:.3g} nm, z {np.std(z) * 1e3:.3g} mm")
        np.savez(os.path.join(a.out, "bhabha_injected.npz"), t=t, x=x, y=y, z=z, ele=ele, pos=pos,
                 ph=ph, ph_evt=ph_evt, ph_t=t[ph_evt], ph_x=x[ph_evt], ph_y=y[ph_evt], ph_z=z[ph_evt],
                 weight=w, L_bx=L_bx, xsec_nb=xs)
    merged = os.path.join(a.out, "inputs_bhabha.txt")
    if rank0:
        write_inputs(read_inputs(a.inputs), overrides, merged)

    if a.check:
        try:
            warpx, callbacks, pcs, libwarpx = warpx_api()
            print("[check] pywarpx found:", warpx.__class__.__module__)
            print("[check] warpx.load_inputs_file:", hasattr(warpx, "load_inputs_file"))
            print("[check] callbacks.installbeforestep:", hasattr(callbacks, "installbeforestep"))
            print("[check] add_particles doc:\n", pcs.ParticleContainerWrapper.add_particles.__doc__)
            print("[check] add_particles takes proper velocity u = gamma*v [m/s]: this driver passes "
                  "u = p / m_e in SI, and unique_particles=False so every rank passes the same events once")
            print("[check] (the compiled WarpX module loads only after warpx.init(); not needed here)")
        except Exception as e:  # noqa: BLE001
            print("[check] pywarpx import failed:", repr(e))
        if "dt" in c:
            print(f"[check] warpx.const_dt from the inputs: {c['dt']:.4g} s; vertex times span "
                  f"steps {int(t.min() / c['dt'])} to {int(t.max() / c['dt'])} of {int(c.get('nt', 0))}")
        print(f"[check] wrote {merged} and bhabha_injected.npz; no WarpX run")
        return

    # ---------------------------------------------------------------- run
    warpx, callbacks, pcs, libwarpx = warpx_api()
    # Every rank builds the same inputs, so each loads its own copy: no rank ever reads a file another
    # rank is still writing (or a stale one from an earlier attempt). Rank 0's copy stays in the run dir.
    if rank0:
        own = merged
    else:
        import tempfile
        own = os.path.join(tempfile.gettempdir(),
                           f"inputs_bhabha.{os.environ.get('SLURM_JOB_ID', 'nojob')}.r{os.environ.get('SLURM_PROCID')}.txt")
        write_inputs(read_inputs(a.inputs), overrides, own)
    warpx.load_inputs_file(own)
    order = np.argsort(t)
    t_s, cursor = t[order], {"i": 0, "n_added": 0, "step": 0}
    uxe, uye, uze = u_from_p(ele[order]); uxp, uyp, uzp = u_from_p(pos[order])
    xs_, ys_, zs_ = x[order], y[order], z[order]
    wts = (order + 1).astype(np.float64) if a.tag_weights else np.full(len(t_s), w)
    if rank0 and a.tag_weights:
        print("[wx_bhabha] --tag-weights: test-particle weight = event index + 1 (physical weight "
              f"{w:.4g} is in bhabha_injected.npz)")

    def inject():
        if "tilt" not in cursor:
            # the vertex sampler assumes beam1's slices sit at x = +tan(phi) * s; check against WarpX's
            # own beam1 and mirror the vertices in x if WarpX tilts the other way
            sgn = beam1_tilt_sign(pcs)
            cursor["tilt"] = sgn
            if sgn < 0:
                xs_[:] = -xs_
            if rank0:
                print(f"[wx_bhabha] beam1 tilt sign from WarpX: {sgn:+.0f}"
                      + (" -> vertices mirrored in x" if sgn < 0 else " (matches the sampler)"))
        try:
            tnow, dt = sim_time_dt(libwarpx)
            src = "WarpX"
        except Exception:  # noqa: BLE001  -- fall back to counting steps of the constant dt
            dt = c["dt"]
            tnow, src = cursor["step"] * dt, "step count x warpx.const_dt"
        if "clock" not in cursor:
            cursor["clock"] = src
            if rank0:
                print(f"[wx_bhabha] simulation time from {src}; dt = {dt:.4g} s")
        cursor["step"] += 1
        i0 = cursor["i"]
        i1 = int(np.searchsorted(t_s, tnow + dt, side="left"))
        if i1 <= i0:
            return
        sl = slice(i0, i1)
        for name, (ux, uy, uz) in (("ele_bhl", (uxe, uye, uze)), ("pos_bhl", (uxp, uyp, uzp))):
            pcs.ParticleContainerWrapper(name).add_particles(
                x=xs_[sl], y=ys_[sl], z=zs_[sl], ux=ux[sl], uy=uy[sl], uz=uz[sl], w=wts[sl],
                unique_particles=False)
        cursor["i"] = i1; cursor["n_added"] += i1 - i0

    callbacks.installbeforestep(inject)
    warpx.init()
    warpx.evolve()
    if rank0:
        late = len(t_s) - cursor["i"]
        print(f"[wx_bhabha] injected {cursor['n_added']} events"
              + (f"; {late} had vertex times after the end of the run" if late else ""))
    warpx.finalize()


if __name__ == "__main__":
    main()
