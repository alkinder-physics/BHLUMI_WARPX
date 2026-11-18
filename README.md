# BHLUMI + WarpX on NERSC Perlmutter

Injects small-angle Bhabha events from [BHLUMI](https://github.com/KrakowHEPSoft/BHLUMI) 4.x into a WarpX
FCC-ee beam-beam simulation and tracks the outgoing e⁺e⁻ through the beam fields.

| File | Purpose |
| --- | --- |
| `setup_bhlumi.sh` | clones, patches and builds BHLUMI in `$HOME/src/bhlumi`; writes the FCC-ee Z card `fcc_z.data` and `run_bhlumi.sh` there |
| `lhe_to_npz.py` | converts BHLUMI's `out.lhe` to `events.npz` |
| `wx_bhabha.py` | runs WarpX through `pywarpx` and injects the events at sampled vertices |
| `submit_wx_bhabha.sh` | Slurm job for one run on Perlmutter GPUs |

Requirements: `gfortran` (`PrgEnv-gnu`), `python3` with NumPy, and a WarpX GPU build with `pywarpx`.

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

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `gfortran not found` | `module load PrgEnv-gnu` |
| `ERROR: build failed` | see `$HOME/src/bhlumi/build.log` |
| `ERROR: no seed file` | pick a seed listed in the error |
| `Could not find the geometry.dims` on ranks ≠ 0 | old `wx_bhabha.py`; pull the current version |

## Reference

S. Jadach et al., *BHLUMI version 4.04*, Comput. Phys. Commun. **102** (1997) 229.
