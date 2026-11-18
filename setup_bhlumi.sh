#!/bin/bash
#
# setup_bhlumi.sh -- install BHLUMI 4.x in $HOME/src/bhlumi on Perlmutter, patched for FCC-ee,
#                    and run a 1000-event test. Self-contained: needs only this file.
#
#     bash setup_bhlumi.sh          (on a login node)
#
# What it changes in BHLUMI (github.com/KrakowHEPSoft/BHLUMI):
#   400/bhlum4.f    WRITE_LHE summed 5 momentum components into a 4-element array (DO J=1,5 -> 1,4)
#   4.x-cpc/demo2.f writes the sample's final cross section to xsec.txt (needed to weight events)
# It also writes:
#   $HOME/src/bhlumi/fcc_z.data     input card: 91.2 GeV, BHLUM4 constant weights, 40-120 mrad
#   $HOME/src/bhlumi/run_bhlumi.sh  one event sample per seed into $PSCRATCH/bhlumi/fcc_z/seed_<n>

set -euo pipefail
DEST=$HOME/src/bhlumi
REPO=${BHLUMI_REPO:-https://github.com/KrakowHEPSoft/BHLUMI.git}

command -v gfortran >/dev/null || module load PrgEnv-gnu
command -v gfortran >/dev/null || { echo "ERROR: gfortran not found (try: module load PrgEnv-gnu)"; exit 1; }

# ---------------------------------------------------------------- 1. source
if [[ -d "$DEST/4.x-cpc" ]]; then
    echo "BHLUMI source already in $DEST"
else
    mkdir -p "$DEST"
    [[ -z "$(ls -A "$DEST")" ]] || { echo "ERROR: $DEST is not empty and has no BHLUMI source"; exit 1; }
    git clone -q "$REPO" "$DEST"
    echo "cloned BHLUMI into $DEST"
fi
cd "$DEST"

# ---------------------------------------------------------------- 2. patch (idempotent)
python3 - <<'PY'
import re
p = "400/bhlum4.f"
s = open(p).read()
i = s.index("SUBROUTINE WRITE_LHE")
j = s.find("SUBROUTINE", i + 20)
j = len(s) if j < 0 else j
body = s[i:j]
if "DO J=1,5" in body:
    s = s[:i] + body.replace("DO J=1,5", "DO J=1,4") + s[j:]
    open(p, "w").write(s); print("patched", p)
else:
    print(p, "already patched")

p = "4.x-cpc/demo2.f"
s = open(p).read()
if "xsec.txt" in s:
    print(p, "already patched")
else:
    anchor = " 300  CONTINUE"
    k = s.index(anchor)
    m = s.index("CALL  DUMPEH(IEV)", k)
    e = s.index("\n", m) + 1
    add = ("!     Final MC cross section of the generated sample [nb] -> xsec.txt\n"
           "      CALL BHLUMI(   2,XPAR,NPAR)\n"
           "      OPEN(78, FILE='xsec.txt')\n"
           "      WRITE(78,'(A,I12)')      'NEVGEN   ', NPAR(10)\n"
           "      WRITE(78,'(A,1PE20.10)') 'XSEC_NB  ', XPAR(10)\n"
           "      WRITE(78,'(A,1PE20.10)') 'REL_ERR  ', XPAR(11)\n"
           "      CLOSE(78)\n")
    s = s[:e] + add + s[e:]
    open(p, "w").write(s); print("patched", p)
PY

# ---------------------------------------------------------------- 3. input card
cat > "$DEST/fcc_z.data" <<'CARD'
fccz  fccz   ======= FCC-ee Z pole, BHLUM4, constant weights ======
 0 0 0 0 0 0 0 0    KAT1,KAT2,KAT3,KAT4,KAT5,KAT6,KAT7,KAT8
     20000      NEVT
      3101      KEYOPT = 1000*KEYGEN +100*KEYREM +10*KEYWGT +KEYRND
        20      KEYRAD =              100*KEYUPD +10*KEYMOD +KEYPIA
         2      KEYTRI  Obsolete!!!
    91.2D0      CMSENE
      .040      Tming   theta_min [rad]    generation
      .120      Tmaxg   theta_max          generation
  0.9999D0      VMAXG   v_max              generation
      1D-3      XK0     eps_CMS            generation
      .062      TminW   theta_min          trigger wide
      .088      TmaxW   theta_max          trigger wide
      .064      TminN   theta_min          trigger narrow
      .086      TminN   theta_max          trigger narrow
     0.5D0      VMAXE   v_max              trigger maximum v
        32      NPHI    nphi               trigger no of phi   sect.
        16      NTHE    ntheta             trigger no of theta sect.
================= the end of data set ================
CARD

# ---------------------------------------------------------------- 4. run script
cat > "$DEST/run_bhlumi.sh" <<'RUN'
#!/bin/bash
# run_bhlumi.sh -- one BHLUMI event sample (LHE) per seed
# Usage:   bash run_bhlumi.sh <seed> <n_events> [card]      e.g. bash run_bhlumi.sh 2 1000000
# Output:  $PSCRATCH/bhlumi/fcc_z/seed_<seed>/out.lhe (events, constant weight) and xsec.txt
# Each event represents XSEC_NB / NEVGEN nb. BHLUMI: incoming e+ along +z, e- along -z.
set -euo pipefail
SEED=${1:?usage: bash run_bhlumi.sh <seed> <n_events> [card]}
NEV=${2:?usage: bash run_bhlumi.sh <seed> <n_events> [card]}
BHL=$HOME/src/bhlumi/4.x-cpc
CARD=${3:-$HOME/src/bhlumi/fcc_z.data}
OUTROOT=$PSCRATCH/bhlumi/fcc_z
[[ -f "$BHL/iniseed/iniseed.$SEED" ]] || { echo "ERROR: no seed file $BHL/iniseed/iniseed.$SEED"; \
    echo "Available: $(ls $BHL/iniseed | sed 's/iniseed\.//' | sort -n | tr '\n' ' ')"; exit 1; }
RUNDIR=$OUTROOT/seed_$SEED
mkdir -p "$RUNDIR"; cd "$RUNDIR"
cp "$BHL/iniseed/iniseed.$SEED" ./iniseed
cp "$BHL/semaphore.start" ./semaphore
awk -v n="$NEV" 'NR==3 {printf "%10d      NEVT\n", n; next} {print}' "$CARD" > card.data
echo "BHLUMI seed $SEED, $NEV events -> $RUNDIR"
time "$BHL/demo2.exe" < card.data > run.log 2>&1
cat xsec.txt
echo "events written: $(grep -c '<event>' out.lhe)"
RUN

# ---------------------------------------------------------------- 5. build
( cd 400 && make -s ) > build.log 2>&1
( cd lib && make -s ) >> build.log 2>&1
( cd 4.x-cpc && make -s demo2.exe ) >> build.log 2>&1
[[ -x 4.x-cpc/demo2.exe ]] || { echo "ERROR: build failed, see $DEST/build.log"; exit 1; }
echo "built $DEST/4.x-cpc/demo2.exe"

# ---------------------------------------------------------------- 6. 1000-event test
bash "$DEST/run_bhlumi.sh" 1 1000
echo
echo "Done. Production: bash $DEST/run_bhlumi.sh <seed> <n_events>   (e.g. 2 1000000)"
