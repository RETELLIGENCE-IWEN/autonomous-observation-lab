#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_directory}/open_gimbal_dashboard.sh" recovery \
    --recovery-results artifacts/gimbal_recovery_robustness_fresh_test.json \
    --recovery-scenario target_reversal_outage \
    --seed 45000
