#!/bin/bash
# ---------------------------------------------------------------------------
# The five laptop baselines of Table 1 on the test recordings of folds 2 and 3
# (src/folds.py), with the settings behind the paper's rows. None of them
# trains on this data, so one run serves both folds. A recording that already
# has a stack is skipped: this never overwrites a paper prediction, and
# rerunning it resumes where it stopped.
#
#   bash benchmark/run_baselines_folds.sh                       # all five
#   bash benchmark/run_baselines_folds.sh usam_single omnipose  # some
#
# The SAM 3 rows run on the cluster: slurm/sam3_baselines_folds.sbatch.
#
# Checked 2026-10-07 by rerunning each one on A71_C1 against the paper's stack:
# Cellpose-SAM and Omnipose reproduce it bit for bit, NeuroQuantify to 0.9993
# pixel IoU. uSAM is deterministic and differs only on the 7 A71 frames whose
# GT was corrected after its stack was made (it is prompted with GT boxes).
# Zero-shot SAM 2 is not deterministic on MPS (two runs agree to 0.990 pixel
# IoU, 0.0003 in score), and today's code scores A71 0.756 where the July
# stack, made before run_ours.py was in git, scored 0.760.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE
OMNIPOSE_PY=${OMNIPOSE_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/omnipose/bin/python}
VIDEOS=$(venv/bin/python -c "from src.folds import FOLDS; print(' '.join(v for f in (2, 3) for v in FOLDS[f]))")
METHODS=${*:-sam2_zeroshot usam_single cellpose_sam omnipose neuroquantify}

todo() {    # the recordings of $VIDEOS with no stack in predictions/$1 yet
    local out=""
    for v in $VIDEOS; do
        [ -f "benchmark/predictions/$1/$v.tif" ] || out="$out $v"
    done
    echo $out
}

for m in $METHODS; do
    V=$(todo "$m")
    echo; echo "##### $m  $(date +%T)  ${V:-nothing to do}"
    [ -z "$V" ] && continue
    case "$m" in
        sam2_zeroshot) venv/bin/python benchmark/run_ours.py --no-lora --videos $V \
                           --out-dir benchmark/predictions/sam2_zeroshot ;;
        usam_single)   benchmark/competitors/usam_venv/bin/python benchmark/run_usam.py \
                           --single-mask --videos $V --out-dir benchmark/predictions/usam_single ;;
        cellpose_sam)  benchmark/competitors/cp_venv/bin/python benchmark/run_cellpose.py --videos $V ;;
        omnipose)      "$OMNIPOSE_PY" benchmark/run_omnipose.py --model bact_phase_omni \
                           --scale 3.0 --videos $V ;;
        neuroquantify) benchmark/competitors/neuroquantify_venv/bin/python \
                           benchmark/run_neuroquantify.py --videos $V ;;
        *) echo "unknown method: $m"; exit 2 ;;
    esac || echo "!!! $m failed -- rerun this script to retry the missing recordings"
done
echo; echo "##### done $(date +%T)"
