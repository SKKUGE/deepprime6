#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "====================================================="
echo " Starting Full pegRNA Design Pipeline (Phase 1 ~ 12) "
echo "====================================================="

echo ""
echo "[1/3] Running Phase 1-5 (Data Preprocessing & Design)..."
bash "$SCRIPT_DIR/run_pilot_1_to_5.sh"

echo ""
echo "[2/3] Running Phase 6-8 (Deep Learning Model Scoring)..."
bash "$SCRIPT_DIR/run_pilot_6_to_8.sh"

echo ""
echo "[3/3] Running Phase 9-12 (Normalization & Candidate Selection)..."
bash "$SCRIPT_DIR/run_pilot_9_to_12.sh"

echo ""
echo "====================================================="
echo " All phases completed successfully!                  "
echo "====================================================="
