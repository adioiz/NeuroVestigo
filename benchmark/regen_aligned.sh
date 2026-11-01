#!/bin/bash
# ---------------------------------------------------------------------------
# Re-predict every method whose output was shrunk back to native size
# off-centre, with the centred versions, and rebuild the 3-fold tables.
#
#   bash benchmark/regen_aligned.sh        # about 3-4 hours on the laptop
#
#   NeuroVestigo (the topology checkpoint the paper reports, all three runs)
#   and zero-shot SAM 2:
#       run_ours.py now resizes the logits bilinearly, then thresholds.
#   Cellpose-SAM, Omnipose, NeuroQuantify:
#       target chosen on the upscaled grid, then shrunk by area (native_res.py).
#   uSAM, SAM 3 and SAM 3.1 already return native-size masks and are not rerun.
#
# Each old stack (and its _raw twin) is MOVED to
# benchmark/predictions/_pre_alignment_2026-11/ before its replacement is
# written; a moved stack is never overwritten. So the draft's numbers stay
# reproducible, and rerunning this resumes where it stopped. The baselines run
# first; NeuroVestigo needs run 3's weights from the cluster and is skipped,
# untouched, if they cannot be fetched.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE
PY=venv/bin/python
OMNIPOSE_PY=${OMNIPOSE_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/omnipose/bin/python}
P=benchmark/predictions
BK=$P/_pre_alignment_2026-11
F1="A71_C1 A81_C1 A12_C3_100k A12_C1_100k_2"
F2="A51_C1 A61_C1 B11_C1_100k B12_C1_100k"
F3="A11_C1 A52_C1_right A11_C1_100k A11_C5_100k"
ALL="$F1 $F2 $F3"

grep -q "def logits_to_native" benchmark/run_ours.py \
    || { echo "benchmark/run_ours.py does not have the bilinear fix"; exit 1; }
grep -q "target_at_native" benchmark/run_cellpose.py benchmark/run_omnipose.py benchmark/run_neuroquantify.py \
    || { echo "the automatic runners do not have the area fix"; exit 1; }

redo() {    # redo <dir> "<recordings>" <command...>  ->  <command> --videos <todo> --out-dir <dir>
    local d=$1 recs=$2; shift 2
    local todo=""
    mkdir -p "$BK/$d" "$P/$d"
    for v in $recs; do
        # a stack with no backup yet is an old one: move it (and its _raw twin) aside
        if [ -f "$P/$d/$v.tif" ] && [ ! -f "$BK/$d/$v.tif" ]; then
            mv "$P/$d/$v.tif" "$BK/$d/$v.tif"
            if [ -f "$P/${d}_raw/$v.tif" ]; then
                mkdir -p "$BK/${d}_raw"; mv "$P/${d}_raw/$v.tif" "$BK/${d}_raw/$v.tif"
            fi
        fi
        [ -f "$P/$d/$v.tif" ] || todo="$todo $v"
    done
    echo; echo "##### $d: ${todo:-done already}  ($(date +%T))"
    [ -z "$todo" ] && return 0
    "$@" --videos $todo --out-dir "$P/$d" \
        || { echo "!!! $d failed -- rerun this script to continue"; exit 1; }
}

redo sam2_zeroshot   "$ALL" $PY benchmark/run_ours.py --no-lora
redo cellpose_sam    "$ALL" benchmark/competitors/cp_venv/bin/python benchmark/run_cellpose.py
redo omnipose        "$ALL" "$OMNIPOSE_PY" benchmark/run_omnipose.py --model bact_phase_omni --scale 3.0
redo neuroquantify   "$ALL" benchmark/competitors/neuroquantify_venv/bin/python benchmark/run_neuroquantify.py

# NeuroVestigo last: run 3's weights are on the cluster. Without them, stop here --
# before any NeuroVestigo stack moves and before the tables are rebuilt, so the
# results never mix runs; rerunning resumes at this point.
if [ ! -f checkpoints-fold3/lora_weights_topo.pt ]; then
    echo "fetching run 3's weights from the cluster ..."
    rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
        --exclude train_state.pt --exclude val_samples \
        slurm-login:Fine-tune-SAM2-model-for-Neurons-segmentation/checkpoints-fold3/ checkpoints-fold3/ \
        && [ -f checkpoints-fold3/lora_weights_topo.pt ] \
        || { echo "cannot fetch run 3's weights (connect to the cluster); the baselines are done, NeuroVestigo is untouched; rerun this script"; exit 1; }
fi
redo ours_23rec_topo "$F1"  $PY benchmark/run_ours.py --ckpt checkpoints-23rec/lora_weights_topo.pt
redo ours_fold2_topo "$F2"  $PY benchmark/run_ours.py --ckpt checkpoints-fold2/lora_weights_topo.pt
redo ours_fold3_topo "$F3"  $PY benchmark/run_ours.py --ckpt checkpoints-fold3/lora_weights_topo.pt

echo; echo "##### 3-fold tables with centred resizes  ($(date +%T))"
$PY benchmark/aggregate_folds.py
echo; echo "Old tables: git show HEAD:benchmark/results/folds/table1.csv"
