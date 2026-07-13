#!/usr/bin/env bash
# run_rollout.sh — Launch ViTacACT real-robot rollout
#
# Usage
# -----
#   ./run_rollout.sh [options]
#
# Options
# -------
#   --config     PATH    Rollout YAML config file (default: rollout/configs/rollout.yml)
#   --task       NAME    Task name: insert_tube | wipe_board (overrides task_id in config)
#   --task-id    INT     Task ID integer (alternative to --task; cannot use both)
#   --duration-s FLOAT   Run duration in seconds (default: unlimited, stop with Ctrl-C)
#   --skip-homing        Skip the homing step (useful for debugging)
#   --dry-run            Print resolved parameters without executing anything
#
# Examples
# --------
#   ./run_rollout.sh --task insert_tube
#   ./run_rollout.sh --task wipe_board --duration-s 60
#   ./run_rollout.sh --config my_config.yml --task insert_tube --skip-homing

set -euo pipefail

# ── Script location ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────
CONFIG="${SCRIPT_DIR}/rollout/configs/rollout.yml"
TASK=""
TASK_ID=""
DURATION_S=""
SKIP_HOMING=0
DRY_RUN=0

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";      shift 2 ;;
        --task)         TASK="$2";        shift 2 ;;
        --task-id)      TASK_ID="$2";     shift 2 ;;
        --duration-s)   DURATION_S="$2";  shift 2 ;;
        --skip-homing)  SKIP_HOMING=1;    shift   ;;
        --dry-run)      DRY_RUN=1;        shift   ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: $0 [--config PATH] [--task NAME] [--task-id INT] [--duration-s FLOAT] [--skip-homing] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ -n "$TASK" && -n "$TASK_ID" ]]; then
    echo "[ERROR] --task and --task-id cannot be used together." >&2
    exit 1
fi

# ── Parameter summary ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  Flexiv ViTacACT Rollout"
echo "============================================================"
echo "  config     : $CONFIG"
if [[ -n "$TASK" ]]; then
    echo "  task       : $TASK"
elif [[ -n "$TASK_ID" ]]; then
    echo "  task       : (id=$TASK_ID)"
else
    echo "  task       : (not specified, using config default)"
fi
echo "  duration   : ${DURATION_S:-unlimited (Ctrl-C to stop)}"
echo "  homing     : $([ $SKIP_HOMING -eq 1 ] && echo 'skipped' || echo 'enabled (robot id=2)')"
echo "============================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Parameters look good. Exiting without execution."
    exit 0
fi

# ── Step 1: Homing ─────────────────────────────────────────────────────────────
if [[ $SKIP_HOMING -eq 0 ]]; then
    echo ""
    echo "[homing] Homing robot id=2 (Rizon4s-063586) ..."
    python "${SCRIPT_DIR}/utils/homing.py" --id 2
    echo "[homing] Homing complete."
fi

# ── Step 2: Build run_rollout.py command ───────────────────────────────────────
CMD=(python "${SCRIPT_DIR}/rollout/run_rollout.py" --config "$CONFIG")

if [[ -n "$TASK" ]]; then
    CMD+=(--task "$TASK")
elif [[ -n "$TASK_ID" ]]; then
    CMD+=(--task-id "$TASK_ID")
fi

if [[ -n "$DURATION_S" ]]; then
    CMD+=(--duration-s "$DURATION_S")
fi

# ── Step 3: Launch rollout ─────────────────────────────────────────────────────
echo ""
echo "[rollout] Command: ${CMD[*]}"
echo "------------------------------------------------------------"
exec "${CMD[@]}"
