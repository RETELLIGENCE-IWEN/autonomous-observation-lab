#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/.." && pwd)"
dashboard="${1:-closed-loop}"

case "${dashboard}" in
    closed-loop|causality|benchmark-suite)
        ;;
    *)
        dashboard="closed-loop"
        ;;
esac

cd -- "${repository_root}"
export PYTHONPATH="${repository_root}/src"
exec python3 -m autonomous_observation_lab.gimbal_servoing.visualization \
    --demo "${dashboard}" \
    --spawn
