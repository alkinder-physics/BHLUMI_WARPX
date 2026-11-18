#!/bin/bash -l
#
# submit_wx_bhabha.sh -- one WarpX beam-beam run with BHLUMI Bhabha events injected (pywarpx).
#
#   sbatch submit_wx_bhabha.sh                                   # defaults below
#   sbatch --export=ALL,SEED=2,BZ=2,NINJ=20000 submit_wx_bhabha.sh
#   sbatch -q debug -t 00:15:00 --export=ALL,NINJ=2000 submit_wx_bhabha.sh     # quick test
#   sbatch --export=ALL,EXTRA="--set ele_bhl.do_not_gather=1 --set pos_bhl.do_not_gather=1",TAG=nofield \
#          submit_wx_bhabha.sh                                   # validation: no beam fields on the Bhabhas
#
# Output: $PSCRATCH/runs/warpx/FCC_bhabha/B_<BZ>T[_<TAG>]/seed_<SEED>/
#         output.txt, inputs_bhabha.txt, bhabha_injected.npz, diags/ (ele_bhl, pos_bhl in particles_in
#         and at every wall)

#SBATCH -A m4272
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --exclusive
#SBATCH -J wx_bhabha
#SBATCH -o wx_bhabha_%j.out
#SBATCH -e wx_bhabha_%j.err

set -euo pipefail
SEED=${SEED:-1}
BZ=${BZ:-0}
NINJ=${NINJ:-20000}
EVENTS=${EVENTS:-$PSCRATCH/bhlumi/fcc_z/seed_2/events.npz}
INPUTS=${INPUTS:-$HOME/projects/FCC_pT_plots/wx_tier1_template.txt}
TOOLS=${TOOLS:-$HOME/src/bhlumi-warpx}
TAG=${TAG:-}
EXTRA=${EXTRA:-}

RUNDIR=$PSCRATCH/runs/warpx/FCC_bhabha/B_${BZ}T${TAG:+_$TAG}/seed_${SEED}
mkdir -p "$RUNDIR"
cd "$RUNDIR"
cp "$INPUTS" ./inputs_template.txt

set +u
source "$HOME/perlmutter_gpu_warpx.profile"        # also activates the warpx-gpu venv with pywarpx
set -u
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1                           # keep the [wx_bhabha] lines even if a rank crashes
export AMREX_DEFAULT_INIT="amrex.use_gpu_aware_mpi=1"

echo "Bhabha run: seed=$SEED Bz=$BZ T n_inject=$NINJ events=$EVENTS extra='$EXTRA' dir=$RUNDIR"
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    python3 $TOOLS/wx_bhabha.py --inputs ./inputs_template.txt --events $EVENTS \
        --n-inject $NINJ --seed $SEED --out . \
        --set my_constants.Bz=$BZ --set warpx.random_seed=$SEED $EXTRA" \
  > output.txt 2>&1
grep "\[wx_bhabha\]" output.txt || true
echo "Finished. Output in $RUNDIR"
