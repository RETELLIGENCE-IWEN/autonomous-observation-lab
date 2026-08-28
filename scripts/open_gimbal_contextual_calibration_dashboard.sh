#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_directory}/open_gimbal_dashboard.sh" calibration \
    --uncertainty-calibration \
    artifacts/gimbal_o2_contextual_uncertainty_calibration.json
