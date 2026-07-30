#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${ROOT}"

"${ROOT}/validate_static_tdma.sh"
"${ROOT}/build_static_tdma_mpmc.sh"

submission="$(
    sbatch \
        --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" \
        "${PROJECT_ROOT}/run_mpmc_static_tdma_sweep.slurm"
)"

echo
echo "${submission}"
echo "Monitor with: squeue -u ${USER}"
echo "One allocation compares dynamic/static TDMA operators at NP1, NP2,"
echo "and all three NP4 rank placements using the direct C2I data path."
