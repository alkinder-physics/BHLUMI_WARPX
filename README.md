# BHLUMI + WarpX on NERSC Perlmutter

Small-angle Bhabha events from [BHLUMI](https://github.com/KrakowHEPSoft/BHLUMI) for WarpX beam-beam
simulations of FCC-ee, set up on NERSC Perlmutter.

> **Status (2026-11-18):** BHLUMI install and event generation: tested on Perlmutter (login node, PrgEnv-gnu).
> WarpX coupling (version 1): passes `--check` on Perlmutter. The first debug run hit an MPI-rank race when
> reading the inputs file (fixed in this version); the rerun is pending
> (see [Coupling to WarpX](#coupling-to-warpx)).

## Repository layout

| File | Purpose |
| --- | --- |
| `setup_bhlumi.sh` | installs, patches and builds BHLUMI in `$HOME/src/bhlumi`; writes `fcc_z.data` and `run_bhlumi.sh` there |
| `lhe_to_npz.py` | converts a BHLUMI `out.lhe` to a compact `events.npz` |
| `wx_bhabha.py` | runs WarpX through `pywarpx` with the BHLUMI events injected at their vertices |
| `submit_wx_bhabha.sh` | Slurm script for one `wx_bhabha.py` run on Perlmutter GPUs |

Clone this repository to `$HOME/src/bhlumi-warpx` (the path `submit_wx_bhabha.sh` expects):

```bash
git clone https://github.com/alkinder-physics/BHLUMI_WARPX.git $HOME/src/bhlumi-warpx
```

## Contents

- [Repository layout](#repository-layout)
- [What this does](#what-this-does)
- [Requirements](#requirements)
- [Install BHLUMI](#install-bhlumi)
- [Generate events](#generate-events)
- [Output files](#output-files)
- [Coupling to WarpX](#coupling-to-warpx)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## What this does

BHLUMI generates small-angle Bhabha scattering (e⁺e⁻ → e⁺e⁻ + nγ) with multiphoton radiative corrections.
It is the generator used for luminometer (LumiCal) studies. Guinea-Pig can carry BHLUMI events through the
beam-beam collision; this repository does the same for WarpX:

1. Install and patch BHLUMI on Perlmutter (`setup_bhlumi.sh`).
2. Generate event samples at the FCC-ee Z pole (`run_bhlumi.sh`).
3. Convert the events (`lhe_to_npz.py`), inject them into a WarpX beam-beam run and track the outgoing e±
   through the beam fields (`wx_bhabha.py`, `submit_wx_bhabha.sh`).

## Requirements

- A NERSC account with access to Perlmutter.
- `gfortran` (in the default `PrgEnv-gnu` environment; the script loads it if needed).
- `git` and `python3` (both available on the login nodes).
- For the WarpX step: a WarpX GPU build with the Python bindings (`pywarpx`), following the
  [WarpX Perlmutter instructions](https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html).

## Install BHLUMI

`setup_bhlumi.sh` is self-contained. On a Perlmutter **login node**:

```bash
bash $HOME/src/bhlumi-warpx/setup_bhlumi.sh
```

It:

1. clones BHLUMI into `$HOME/src/bhlumi` (an existing empty folder is fine);
2. applies two small source changes (below);
3. writes the FCC-ee Z input card `fcc_z.data` and the run script `run_bhlumi.sh` into `$HOME/src/bhlumi`;
4. builds `4.x-cpc/demo2.exe` (build log in `$HOME/src/bhlumi/build.log`);
5. runs a 1000-event test into `$PSCRATCH/bhlumi/fcc_z/seed_1`.

A successful install prints (output from Perlmutter, 2026-11-18; timing lines omitted):

```
cloned BHLUMI into /global/homes/a/<user>/src/bhlumi
patched 400/bhlum4.f
patched 4.x-cpc/demo2.f
built /global/homes/a/<user>/src/bhlumi/4.x-cpc/demo2.exe
BHLUMI seed 1, 1000 events -> /pscratch/sd/a/<user>/bhlumi/fcc_z/seed_1
NEVGEN           1000
XSEC_NB      7.0518438895E+01
REL_ERR      9.0644676799E-03
events written: 1000
```

Seed 1 is deterministic, so the test reproduces `XSEC_NB = 70.518 nb` exactly; a different number means
the build or the card differs. The whole install takes under a minute.

The script is safe to rerun: it skips the clone and the source changes when they are already done.

### Source changes

| File | Change | Why |
| --- | --- | --- |
| `400/bhlum4.f` | in `WRITE_LHE`, `DO J=1,5` → `DO J=1,4` (twice) | the loop summed five momentum components into the four-element array `SUM(4)` |
| `4.x-cpc/demo2.f` | after the event loop, call `BHLUMI(2,XPAR,NPAR)` and write `xsec.txt` | the sample's final cross section is needed to give each event its weight |

### Where things live

| Path | Contents | Note |
| --- | --- | --- |
| `$HOME/src/bhlumi` | source, build, `fcc_z.data`, `run_bhlumi.sh` | backed up, small |
| `$PSCRATCH/bhlumi/fcc_z/seed_<n>` | event samples | scratch is purged after 8 weeks: copy samples you keep to CFS |

## Generate events

```bash
bash $HOME/src/bhlumi/run_bhlumi.sh <seed> <n_events> [card]
# e.g.
bash $HOME/src/bhlumi/run_bhlumi.sh 2 1000000
```

- `<seed>` picks one of BHLUMI's random-seed files in `4.x-cpc/iniseed/`; use a different seed for each
  independent sample.
- For more than about 10⁵ events, run on a compute node rather than the login node:

```bash
salloc -q interactive -C cpu -t 00:30:00 -N 1
bash $HOME/src/bhlumi/run_bhlumi.sh 2 1000000
exit
```

Reference production run (Perlmutter CPU node, 2026-11-18): 10⁶ events in 42 s,
`XSEC_NB = 69.451 nb`, `REL_ERR = 3.0e-4`. BHLUMI runs on one core, so independent seeds can run side by
side on one node.

### Events per bunch crossing

The number of Bhabhas per bunch crossing in the generation window is σ × 𝓛_BX, with 𝓛_BX the luminosity
per crossing (take it from the WarpX luminosity diagnostic). For FCC-ee Z with N = 2.02×10¹¹,
σx = 8.0 µm, σy = 35 nm and Piwinski angle Φ ≈ 31, the geometric estimate is

𝓛_BX ≈ N² / (4π σx σy √(1+Φ²)) ≈ 3.7×10²⁸ cm⁻²,

so 69.5 nb gives about 2.6×10⁻³ Bhabhas per crossing between 40 and 120 mrad (one every ~400 crossings).
A WarpX run therefore injects many BHLUMI events, each carrying the weight σ × 𝓛_BX / N_injected.

### Input card `fcc_z.data`

| Setting | Value |
| --- | --- |
| Generator | BHLUM4, constant-weight events (`KEYOPT = 3101`) |
| Centre-of-mass energy | 91.2 GeV |
| Generation window | θ from 40 to 120 mrad |
| Cross section in this window | ≈ 70 nb (1000-event test: 70.5 ± 0.6 nb) |

Change the angular window or energy by editing a copy of the card and passing it as the third argument.

## Output files

Each `seed_<n>/` folder holds:

| File | Contents |
| --- | --- |
| `out.lhe` | events in Les Houches format: final-state e⁺, e⁻ and photons, momenta and energies in GeV (1.6 GB per 10⁶ events) |
| `xsec.txt` | `NEVGEN` (events generated), `XSEC_NB` (cross section of the sample, nb), `REL_ERR` |
| `run.log`, `bhl.output` | BHLUMI's log and summary |

**Event weight.** With constant weights, each event represents `XSEC_NB / NEVGEN` nb. The number of Bhabha
events per bunch crossing is that cross section times the luminosity per crossing.

**Beam direction.** BHLUMI puts the incoming e⁺ along +z and the e⁻ along −z. In our WarpX inputs `beam1`
(e⁻) moves along +z, so events must be mirrored in z before injection.

## Coupling to WarpX

This follows how Guinea-Pig carries BHLUMI events through the collision. Version 1 is complete except for
step 2 (beamstrahlung-degraded energies).

| Step | What `wx_bhabha.py` does |
| --- | --- |
| 1. Vertex and time | samples each event's (t, x, y, z) from the luminosity density 2c·n₁n₂ of the two bunches: Gaussian, tilted by ±15 mrad in the head-on frame, hourglass-focused at z = 0 (β*x = 90 mm, β*y = 0.7 mm). The same integral gives 𝓛_BX. |
| 2. Energies | *not yet*: events keep their nominal 45.6 GeV beams |
| 3. Frame | mirrors z (BHLUMI e⁻ beam along −z → WarpX `beam1` e⁻ along +z). BHLUMI's frame is the WarpX head-on frame otherwise (the beam energies differ by cos φ, a 10⁻⁴ effect). |
| 4. Injection | adds the e⁺ and e⁻ as test species `pos_bhl`, `ele_bhl` (no charge deposition, beam fields gathered), each with weight σ·𝓛_BX/N, in the timestep in which its vertex time falls (`installbeforestep` callback) |
| 5. Output | `ele_bhl`, `pos_bhl` in the `particles_in` dump and in the boundary-scraping diagnostic at every wall; photons (untracked, straight lines) and everything injected in `bhabha_injected.npz` |

In the head-on frame the luminous region is short in z (|z| ≲ σx/2φ ≈ 0.3 mm) and moves across x as
x = ±φ·c(t − t_c): the lab's crossing angle seen from the boosted frame. Which sign depends on WarpX's tilt
convention, so the driver measures the tilt of `beam1` as WarpX initialised it and mirrors the vertices in x
if needed (it prints which).

### Species names: `bh`, `bha` and `bhl` are different things

| Prefix | Mechanism | Made by | Species |
| --- | --- | --- | --- |
| `bh` | Bethe–Heitler pair creation, γ* γ → e⁺e⁻ | WarpX QED collision on the grid | `ele_bh`, `pos_bh` |
| `bha` | radiative Bhabha in WarpX: linear Compton of a virtual photon on a beam particle, γ* e → γ e | WarpX QED collision on the grid | `pho*_bha`, `ele_compt`, `pos_compt` |
| `bhl` | full Bhabha scattering, e⁺e⁻ → e⁺e⁻(nγ), with BHLUMI's radiative corrections | external generator, injected by `wx_bhabha.py` | `ele_bhl`, `pos_bhl` |

Native WarpX products are created where their parents overlap inside a step. BHLUMI events are complete
before the run starts. WarpX only chooses where and when each event happens and then tracks the e⁺ and e⁻.

**Box caveat.** At 40–120 mrad a Bhabha moves c·Δt·θ ≈ 10–31 µm sideways per step, so in the base box
(half-height 13 µm) it leaves within 1–2 steps. Its beam-beam deflection is then truncated. Use a taller
box, or add the far-field kick analytically after the particle exits.

### Run it

1. Convert the events once (writes `events.npz` next to `out.lhe`):

   ```bash
   python3 $HOME/src/bhlumi-warpx/lhe_to_npz.py $PSCRATCH/bhlumi/fcc_z/seed_2
   ```

   Reference (Perlmutter, 2026-11-18):
   `events.npz: 1000000 events (0 skipped), 1605914 photons; xsec = 69.4513 nb from 1000000 generated`.
   Every event must keep exactly one e⁻ and one e⁺; `skipped` counts those that did not.

2. Check the setup on a login node (no WarpX run; writes `inputs_bhabha.txt` and `bhabha_injected.npz`):

   ```bash
   source $HOME/perlmutter_gpu_warpx.profile
   mkdir -p $PSCRATCH/bhabha_check && cd $PSCRATCH/bhabha_check
   python3 $HOME/src/bhlumi-warpx/wx_bhabha.py --check \
       --inputs $HOME/projects/FCC_pT_plots/wx_tier1_template.txt \
       --events $PSCRATCH/bhlumi/fcc_z/seed_2/events.npz --n-inject 2000
   ```

   Reference output (Perlmutter, 2026-11-18, FCC-ee Z template, 10⁶-event sample):

   ```
   [wx_bhabha] L_BX = 3.314e+32 m^-2 (sampled, hourglass + tilt; geometric estimate 3.681e+32)
   [wx_bhabha] sigma = 69.451 nb -> 0.002301 Bhabhas per crossing; 2000 injected, weight 1.151e-06 each
   [wx_bhabha] vertex spread: c(t - t_c) 12 mm, x 180 um, y 27.5 nm, z 0.351 mm
   [check] warpx.load_inputs_file: True
   [check] callbacks.installbeforestep: True
   ```

   𝓛_BX is about 10% under the geometric estimate because of the hourglass effect (β*y = 0.7 mm is
   comparable to the luminous length). `add_particles` takes proper velocity u = γv in m/s, which is what
   the driver passes; with `unique_particles=False` every rank passes the same events and WarpX adds each
   once. The compiled WarpX module only loads after `warpx.init()`, so `--check` does not test it; the
   driver reads the simulation time from WarpX and falls back to step count × `warpx.const_dt`.

3. Quick test run, then production:

   ```bash
   sbatch -q debug -t 00:15:00 --export=ALL,NINJ=2000 $HOME/src/bhlumi-warpx/submit_wx_bhabha.sh
   sbatch --export=ALL,SEED=1,BZ=0,NINJ=20000 $HOME/src/bhlumi-warpx/submit_wx_bhabha.sh
   ```

   Output goes to `$PSCRATCH/runs/warpx/FCC_bhabha/B_<Bz>T/seed_<n>/`.

### Validation plan

1. **No beam fields on the Bhabhas** (`TAG=nofield`, `EXTRA="--set ele_bhl.do_not_gather=1 --set pos_bhl.do_not_gather=1"`):
   the angles at the walls must reproduce BHLUMI's exactly.
2. **Beam fields on:** the mean angular shift should match the expected kick from the oncoming bunch.
3. Then add beamstrahlung-degraded energies (step 2).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `gfortran not found` | `module load PrgEnv-gnu`, then rerun |
| `ERROR: build failed` | read `$HOME/src/bhlumi/build.log` |
| `ERROR: no seed file` | the seed number has no file in `4.x-cpc/iniseed/`; the error lists the available ones |
| `... is not empty and has no BHLUMI source` | `$HOME/src/bhlumi` holds something else: move it aside and rerun |
| clone fails | check GitHub access from the login node (`git ls-remote https://github.com/KrakowHEPSoft/BHLUMI.git`) |
| `Could not find the geometry.dims in the input file './inputs_bhabha.txt'` on ranks ≠ 0 | old `wx_bhabha.py`: other ranks read the inputs file while rank 0 was still writing it. Update the script; each rank now loads its own copy |
| no `[wx_bhabha]` lines in `output.txt` after a crash | old `submit_wx_bhabha.sh` without `PYTHONUNBUFFERED=1`; update it |

## References

- S. Jadach, W. Płaczek, E. Richter-Wąs, B.F.L. Ward, Z. Wąs, *Upgrade of the Monte Carlo program BHLUMI
  for Bhabha scattering at low angles to version 4.04*, Comput. Phys. Commun. **102** (1997) 229.
- BHLUMI source: <https://github.com/KrakowHEPSoft/BHLUMI>
- WarpX: <https://github.com/BLAST-WarpX/warpx>
