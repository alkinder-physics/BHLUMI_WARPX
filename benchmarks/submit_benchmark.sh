#!/bin/bash -l
#
# submit_benchmark.sh -- one analytic-benchmark run: BHLUMI events through round head-on beams (no external
# field, no QED). Each scraped lepton is tagged with its event index (--tag-weights) for the analysis in
# notebooks/01_headon_round_benchmark.ipynb.
#
#   sbatch benchmarks/submit_benchmark.sh                                    # default benchmark input
#   sbatch -q debug -t 00:15:00 --export=ALL,NINJ=5000 benchmarks/submit_benchmark.sh      # quick test
#   sbatch --export=ALL,INPUTS=$PWD/generated_inputs/headon/inputs_npart4e+10.txt,TAG=npart4e10 \
#          benchmarks/submit_benchmark.sh                                    # a file from make_inputs.ipynb
#
# Output: $PSCRATCH/runs/warpx/bhabha_benchmark/<TAG>/seed_<SEED>/
#         output.txt, inputs_template.txt, inputs_bhabha.txt, bhabha_injected.npz, diags/

#SBATCH -A m4272
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --exclusive
#SBATCH -J wx_bhabha_bench
#SBATCH -o wx_bhabha_bench_%j.out
#SBATCH -e wx_bhabha_bench_%j.err

set -euo pipefail
TOOLS=${TOOLS:-$HOME/src/bhlumi-warpx}
SEED=${SEED:-1}
NINJ=${NINJ:-100000}
EVENTS=${EVENTS:-$PSCRATCH/bhlumi/fcc_z/seed_2/events.npz}
INPUTS=${INPUTS:-$TOOLS/benchmarks/headon_round/inputs_headon_round.txt}
TAG=${TAG:-headon_round}
EXTRA=${EXTRA:-}

RUNDIR=$PSCRATCH/runs/warpx/bhabha_benchmark/${TAG}/seed_${SEED}
mkdir -p "$RUNDIR"
cd "$RUNDIR"
cp "$INPUTS" ./inputs_template.txt

set +u
source "$HOME/perlmutter_gpu_warpx.profile"        # also activates the warpx-gpu venv with pywarpx
set -u
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export AMREX_DEFAULT_INIT="amrex.use_gpu_aware_mpi=1"

echo "Benchmark run: tag=$TAG seed=$SEED n_inject=$NINJ inputs=$INPUTS events=$EVENTS extra='$EXTRA' dir=$RUNDIR"
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    python3 $TOOLS/wx_bhabha.py --inputs ./inputs_template.txt --events $EVENTS \
        --n-inject $NINJ --seed $SEED --out . --tag-weights \
        --set warpx.random_seed=$SEED $EXTRA" \
  > output.txt 2>&1
grep "\[wx_bhabha\]" output.txt || true
echo "Finished. Output in $RUNDIR"
