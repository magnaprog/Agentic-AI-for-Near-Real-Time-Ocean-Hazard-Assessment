#!/usr/bin/env bash
# Run the full evaluation pipeline for reproducibility.
#
# This script downloads DART data, runs every tracked-result-producing
# validation/evaluation script, profiles latency, and generates the
# publication figures.
#
# Network-dependent steps (1, 6) may fail without internet access;
# the remaining steps will still execute, and the script exits nonzero
# if any step failed.
#
# Usage:
#     bash scripts/run_full_evaluation.sh
#
# Prerequisites:
#     pip install -e ".[dev,paper]"   # paper extra: matplotlib, cartopy, pandas
#     Internet access (for NDBC DART data download, steps 1 and 6)
#
# Output:
#     results/*.json          - evaluation result files
#     paper/figures/fig*.pdf  - publication-quality figures
#     paper/appendix_f_generated.tex - synthetic pipeline appendix
#
# Reproducibility notes:
#   - results/latency_profile.json is hardware-bound and expected to differ.
#   - physics_validation.json embeds a generated_at wall-clock timestamp;
#     synthetic_timelines.json embeds generated_at and showed tiny
#     floating-point drift in the recorded clean-room run (max 5.2e-15,
#     within a 1e-12 tolerance). Both match on every field except
#     generated_at, but are not byte-identical.
#   - tohoku_agent_trace.json and all_event_agent_traces.json are
#     point-in-time pipeline traces that no current script regenerates;
#     they are excluded from the reproducibility gate.
#   - The detection (all five events), detiding, false-positive,
#     duplicate-sensitivity, ablation, FSM-transition, synthetic-evaluation,
#     and agent_traces artifacts should reproduce byte-identically from the
#     checked-in data. See docs/USER_MANUAL.md section 15.3.
#   - Step 16 figure maps use Cartopy Natural Earth features, which download
#     into Cartopy's cache on first render (one-time network dependency).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

FAILED=0

run_step() {
    local step_num="$1"
    local desc="$2"
    shift 2
    echo "[${step_num}] ${desc}..."
    if "$@"; then
        echo "[${step_num}] Done."
    else
        echo "[${step_num}] FAILED (exit code $?). Continuing..."
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

echo "=========================================="
echo "Agentic AI for Near-Real-Time Ocean Hazard Assessment"
echo "Full Evaluation Pipeline"
echo "=========================================="
echo ""

# Steps 1 and 6 require network access; the validation steps run on the
# checked-in event data in data/ (step 1 refreshes the Tohoku copy).
run_step "1/17" "Downloading Tohoku 2011 DART data" \
    python3 scripts/download_tohoku_dart.py --calibration-days 30

run_step "2/17" "Validating detiding quality" \
    python3 scripts/validate_detiding.py

run_step "3/17" "Running Tohoku retrospective validation" \
    python3 scripts/validate_tohoku.py --sliding-window

run_step "4/17" "Running Chile 2010 retrospective validation" \
    python3 scripts/validate_chile.py --sliding-window

run_step "5/17" "Running Illapel 2015 retrospective validation" \
    python3 scripts/validate_illapel.py --sliding-window

run_step "6/17" "Running false positive evaluation (June 2011)" \
    python3 scripts/run_false_positive_evaluation.py --download

run_step "7/17" "Running Iquique 2014 retrospective validation" \
    python3 scripts/validate_iquique.py --sliding-window

run_step "8/17" "Running Samoa 2009 retrospective validation" \
    python3 scripts/validate_samoa.py --sliding-window

# Steps 9-17 need no network, except step 16's one-time Cartopy
# Natural Earth cache download on a fresh machine.
run_step "9/17" "Evaluating duplicate-timestamp policy sensitivity" \
    python3 scripts/evaluate_duplicate_sensitivity.py

run_step "10/17" "Running ensemble ablation study" \
    python3 scripts/run_ablation.py

run_step "11/17" "Running synthetic detection sensitivity evaluation" \
    python3 scripts/run_synthetic_evaluation.py

run_step "12/17" "Running physics validation" \
    python3 scripts/run_physics_validation.py

run_step "13/17" "Analyzing FSM transitions" \
    python3 scripts/analyze_fsm_transitions.py

run_step "14/17" "Running synthetic end-to-end pipeline (agent traces + appendix)" \
    python3 scripts/run_synthetic_pipeline.py

run_step "15/17" "Profiling pipeline latencies" \
    python3 scripts/profile_latency.py --iterations 50

run_step "16/17" "Generating publication figures" \
    python3 scripts/generate_paper_figures.py

run_step "17/17" "Generating analytical plots" \
    python3 scripts/generate_analytical_plots.py

echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
    echo "Evaluation complete! All steps succeeded."
else
    echo "Evaluation complete with ${FAILED} failed step(s)."
fi
echo "=========================================="
echo ""
echo "Results:"
ls -la results/*.json 2>/dev/null || echo "  (no result files found)"
echo ""
echo "Figures:"
ls -la paper/figures/fig*.pdf 2>/dev/null || echo "  (no figure files found)"

[ "$FAILED" -eq 0 ] || exit 1
