#!/usr/bin/env bash
# run_rollout.sh — 启动 ViTacACT 真机 rollout
#
# 用法
# ----
#   ./run_rollout.sh [选项]
#
# 选项
# ----
#   --config   PATH        rollout YAML 配置文件（默认: rollout/configs/rollout.yml）
#   --task     TASK_NAME   任务名 insert_tube | wipe_board（覆盖 config 中的 task_id）
#   --task-id  INT         任务 ID（--task 的数字形式，两者任选其一）
#   --duration-s FLOAT     运行时长秒数（默认不限，Ctrl-C 结束）
#   --skip-homing          跳过回零步骤（调试用）
#   --dry-run              只打印参数，不实际执行
#
# 示例
# ----
#   ./run_rollout.sh --task insert_tube
#   ./run_rollout.sh --task wipe_board --duration-s 60
#   ./run_rollout.sh --config my_config.yml --task insert_tube --skip-homing

set -euo pipefail

# ── 脚本位置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
CONFIG="${SCRIPT_DIR}/rollout/configs/rollout.yml"
TASK=""
TASK_ID=""
DURATION_S=""
SKIP_HOMING=0
DRY_RUN=0

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";      shift 2 ;;
        --task)         TASK="$2";        shift 2 ;;
        --task-id)      TASK_ID="$2";     shift 2 ;;
        --duration-s)   DURATION_S="$2";  shift 2 ;;
        --skip-homing)  SKIP_HOMING=1;    shift   ;;
        --dry-run)      DRY_RUN=1;        shift   ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            echo "用法: $0 [--config PATH] [--task TASK] [--task-id INT] [--duration-s FLOAT] [--skip-homing] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ── 参数校验 ──────────────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] config 文件不存在: $CONFIG" >&2
    exit 1
fi

if [[ -n "$TASK" && -n "$TASK_ID" ]]; then
    echo "[ERROR] --task 和 --task-id 不能同时指定" >&2
    exit 1
fi

# ── 打印参数摘要 ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Flexiv ViTacACT Rollout"
echo "============================================================"
echo "  config     : $CONFIG"
echo "  task       : ${TASK:-${TASK_ID:+(id=$TASK_ID)|(未指定，使用 config 默认)}}"
echo "  duration   : ${DURATION_S:-无限制 (Ctrl-C 停止)}"
echo "  homing     : $([ $SKIP_HOMING -eq 1 ] && echo '跳过' || echo '执行 (robot id=2)')"
echo "============================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] 参数检查通过，不实际执行。"
    exit 0
fi

# ── Step 1: 机械臂回零 ────────────────────────────────────────────────────────
if [[ $SKIP_HOMING -eq 0 ]]; then
    echo ""
    echo "[homing] 正在对 robot id=2 (Rizon4s-063586) 执行回零 ..."
    python "${SCRIPT_DIR}/utils/homing.py" --id 2
    echo "[homing] 回零完成。"
fi

# ── Step 2: 构造 run_rollout.py 命令 ─────────────────────────────────────────
CMD=(python "${SCRIPT_DIR}/rollout/run_rollout.py" --config "$CONFIG")

if [[ -n "$TASK" ]]; then
    CMD+=(--task "$TASK")
elif [[ -n "$TASK_ID" ]]; then
    CMD+=(--task-id "$TASK_ID")
fi

if [[ -n "$DURATION_S" ]]; then
    CMD+=(--duration-s "$DURATION_S")
fi

# ── Step 3: 启动 rollout ──────────────────────────────────────────────────────
echo ""
echo "[rollout] 启动命令: ${CMD[*]}"
echo "------------------------------------------------------------"
exec "${CMD[@]}"
