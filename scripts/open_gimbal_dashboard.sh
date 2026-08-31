#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/.." && pwd)"
dashboard="${1:-closed-loop}"
if (( $# > 0 )); then
    shift
fi

case "${dashboard}" in
    closed-loop|causality|benchmark-suite|recovery|calibration|replication|performance)
        ;;
    *)
        dashboard="closed-loop"
        ;;
esac

cd -- "${repository_root}"
export PYTHONPATH="${repository_root}/src"
exec python3 -m autonomous_observation_lab.gimbal_servoing.visualization \
    --demo "${dashboard}" \
    --spawn \
    "$@"
