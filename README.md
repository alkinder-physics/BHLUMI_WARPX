# BHLUMI + WarpX on NERSC Perlmutter

Injects small-angle Bhabha events from [BHLUMI](https://github.com/KrakowHEPSoft/BHLUMI) 4.x into a WarpX
FCC-ee beam-beam simulation and tracks the outgoing e⁺e⁻ through the beam fields.

| File | Purpose |
| --- | --- |
| `setup_bhlumi.sh` | clones, patches and builds BHLUMI in `$HOME/src/bhlumi`; writes the FCC-ee Z card `fcc_z.data` and `run_bhlumi.sh` there |
| `lhe_to_npz.py` | converts BHLUMI's `out.lhe` to `events.npz` |
| `wx_bhabha.py` | runs WarpX through `pywarpx` and injects the events at sampled vertices |
| `submit_wx_bhabha.sh` | Slurm job for one run on Perlmutter GPUs |
| `benchmarks/bhabha_theory.py` | analytic prediction, validity checks and run analysis for the head-on benchmark, all read from the input file |
| `benchmarks/make_inputs.py` | writes benchmark inputs and variants of any existing input file |
| `benchmarks/headon_round/inputs_headon_round.txt` | default benchmark input |
| `benchmarks/submit_benchmark.sh` | Slurm job for one benchmark run (no external field) |
| `docs/headon_benchmark_theory.pdf` | two-page theory of the head-on test (LaTeX source alongside) |
| `notebooks/01_headon_round_benchmark.ipynb` | example 1: theory, checks, run commands and comparison |
| `notebooks/make_inputs.ipynb` | generates scans of input files with a manifest |
| `examples/fcc_z_boxscan_template.txt` | FCC-ee Z box-scan input, the base for scans of the real configuration |

Requirements: `gfortran` (`PrgEnv-gnu`), `python3` with NumPy, and a WarpX GPU build with `pywarpx`.
The notebooks also need Jupyter, Matplotlib and `openpmd_api` (for reading WarpX output).

```bash
git clone https://github.com/alkinder-physics/BHLUMI_WARPX.git $HOME/src/bhlumi-warpx
```

## 1. Install BHLUMI (login node)

```bash
bash $HOME/src/bhlumi-warpx/setup_bhlumi.sh
```

The install ends with a 1000-event test, which must print `XSEC_NB 7.0518438895E+01`. Seed 1 is
deterministic, so a different value means the build or the card differs.

It makes two source changes:
- `400/bhlum4.f`: in `WRITE_LHE`, `DO J=1,5` becomes `DO J=1,4`, which fixes an out-of-bounds sum.
- `4.x-cpc/demo2.f`: writes the sample's cross section to `xsec.txt`.

## 2. Generate and convert events

```bash
salloc -q interactive -C cpu -t 00:30:00 -N 1           # 10^6 events take about 1 min
bash $HOME/src/bhlumi/run_bhlumi.sh 2 1000000            # <seed> <n_events>
python3 $HOME/src/bhlumi-warpx/lhe_to_npz.py $PSCRATCH/bhlumi/fcc_z/seed_2
```

Card: 91.2 GeV, θ = 40–120 mrad, constant weights; σ ≈ 69.45 nb. Each `seed_<n>/` folder holds:
- `out.lhe`, about 1.6 GB per 10⁶ events
- `xsec.txt`
- `events.npz`, which holds the final-state e⁻, e⁺ and photon four-momenta in GeV

BHLUMI puts the e⁻ beam along −z.

## 3. Run WarpX

```bash
cd $HOME/src/bhlumi-warpx
python3 wx_bhabha.py --check --inputs <template> --events <events.npz> --n-inject 2000    # no WarpX run
sbatch -q debug -t 00:15:00 --export=ALL,NINJ=2000 submit_wx_bhabha.sh                  # quick test
sbatch --export=ALL,SEED=1,BZ=0,NINJ=20000 submit_wx_bhabha.sh                          # production
```

Environment variables for `submit_wx_bhabha.sh`:
- `SEED`, `BZ`, `NINJ`
- `EVENTS`, `INPUTS`
- `TAG`, which sets the run-folder suffix
- `EXTRA`, which adds `--set key=value` overrides

Output goes to `$PSCRATCH/runs/warpx/FCC_bhabha/B_<BZ>T[_<TAG>]/seed_<SEED>/`.

What `wx_bhabha.py` does:
1. Samples each event's (t, x, y, z) from the overlap n₁n₂ of the two tilted, hourglass-focused Gaussian
   bunches. The same integral gives the luminosity per crossing, 𝓛_BX.
2. Mirrors z, so the e⁻ moves along +z as WarpX `beam1` does, and gives each event the weight σ·𝓛_BX/N.
3. Adds the e⁻ and e⁺ as test species `ele_bhl` and `pos_bhl` (no charge deposition) in the step that
   contains their vertex time. They are saved in `particles_in` and at every wall.
4. Writes everything it injected, including the untracked photons, to `bhabha_injected.npz`.

The species names are distinct on purpose:
- `bhl` is BHLUMI, injected by this code.
- `bh` is WarpX's native Bethe–Heitler pairs.
- `bha` is WarpX's native radiative Bhabha (the γ*e Compton process).

Limits of version 1:
- Events keep their nominal beam energies; there is no beamstrahlung rescaling yet.
- At 40–120 mrad a Bhabha leaves a box of ±13 µm half-height within 1–2 steps, so its beam-beam
  deflection is truncated. Use a taller box.

## 4. Example 1: analytic benchmark (head-on round beams)

Before the FCC-ee configuration, check the injection and tracking against a closed-form result. For round
Gaussian beams (σ_x = σ_y = σ), head-on, rigid, with no hourglass, beamstrahlung or external field, the
deflection of a Bhabha lepton at polar angle θ, averaged over the luminous region, is

    <Δθ>(θ) = − N r_e / (√π γ′ θ σ_z) · ln(1 + θ²σ_z² / (3σ²))

The derivation is in `docs/headon_benchmark_theory.pdf` and in `notebooks/01_headon_round_benchmark.ipynb`. For the default input (45.6 GeV,
N = 2×10¹⁰, σ = 10 µm, σ_z = 1 mm) this is −15.2, −14.5 and −13.2 µrad at 60, 70 and 90 mrad.

The theory reads every number from the WarpX input file, so editing the parameters in the notebook (or the
file itself) changes the input and the prediction together. The notebook also checks that the file meets the
formula's assumptions: round, head-on, disruption < 0.05, no emittance, no QED, no external fields, enough
transverse and longitudinal resolution, a box wide enough that truncating the kick costs < 10⁻³, and a run
long enough for the last vertex.

```bash
cd $HOME/src/bhlumi-warpx
sbatch -q debug -t 00:15:00 --export=ALL,NINJ=5000,TAG=headon_round_test benchmarks/submit_benchmark.sh
sbatch --export=ALL,NINJ=100000 benchmarks/submit_benchmark.sh
```

Output goes to `$PSCRATCH/runs/warpx/bhabha_benchmark/<TAG>/seed_<SEED>/`. The script runs
`wx_bhabha.py --tag-weights`, which sets each test particle's weight to its event index + 1 so that every
lepton found at a wall (or still in the box at the end) can be matched to its BHLUMI event. The physical
weight stays in `bhabha_injected.npz`. Then set `RUN_DIR` in Section 5 of the notebook.

`notebooks/make_inputs.ipynb` writes scans to `generated_inputs/` (git-ignored) with a `manifest.csv`:
benchmark variants with their predicted kick and any failed checks, and variants of an existing input file
(by default the box-scan input without its external solenoid field).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `gfortran not found` | `module load PrgEnv-gnu` |
| `ERROR: build failed` | see `$HOME/src/bhlumi/build.log` |
| `ERROR: no seed file` | pick a seed listed in the error |
| `Could not find the geometry.dims` on ranks ≠ 0 | old `wx_bhabha.py`; pull the current version |
| `CUDA error 700 ... illegal memory access`, profiler stack ending in `DifferentialLuminosity` | race in WarpX's `DifferentialLuminosity` (no stream sync between boxes); run with `CUDA_LAUNCH_BLOCKING=1` or patch WarpX (see below) |
| notebook: `weights are not event tags` | the run was made without `--tag-weights`; use `benchmarks/submit_benchmark.sh` |

### WarpX patch for the `DifferentialLuminosity` crash

In `Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp`, the per-box particle bins are freed while the
luminosity kernel may still be reading them. Add `amrex::Gpu::streamSynchronize();` right after the
`amrex::For( n_independent_pairs, ... );` loop, inside the `MFIter` loop, and rebuild. Until then, add
`CUDA_LAUNCH_BLOCKING=1` to the job's `--export` list (about 1.5x slower).

## Reference

S. Jadach et al., *BHLUMI version 4.04*, Comput. Phys. Commun. **102** (1997) 229.
