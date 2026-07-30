#!/bin/bash
#SBATCH -J PaScaL_STATIC_TDMA
#SBATCH -p batch
#SBATCH --nodelist=gpu01,gpu02
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=01:00:00
#SBATCH -o %x_%j.output
#SBATCH -e %x_%j.err

set -euo pipefail

resolve_project_root()
{
    local candidate

    for candidate in \
        "${PROJECT_ROOT:-}" \
        "${SLURM_SUBMIT_DIR:-}" \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
        if [[ -n "${candidate}" \
              && -x "${candidate}/run/examples_debug_c2i_direct.exe" \
              && -x "${candidate}/run/examples_perf_c2i_direct.exe" \
              && -f "${candidate}/run/PARA_INPUT_4.dat" \
              && -x "${candidate}/summarize_static_tdma_sweep.sh" ]]; then
            cd "${candidate}"
            pwd
            return 0
        fi
    done
    return 1
}

if ! PROJECT_ROOT="$(resolve_project_root)"; then
    echo "[ERROR] Could not locate the built static-TDMA A/B project root."
    echo "[ERROR] PROJECT_ROOT=${PROJECT_ROOT:-unset}"
    echo "[ERROR] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
    exit 2
fi
export PROJECT_ROOT

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/load_mpmc_env.sh"

CUDECOMP_INSTALL="${PROJECT_ROOT}/_install/cudecomp"
export LD_LIBRARY_PATH="${CUDECOMP_INSTALL}/lib:${LD_LIBRARY_PATH}"

DEBUG_DIRECT="${PROJECT_ROOT}/run/examples_debug_c2i_direct.exe"
PERF_DIRECT="${PROJECT_ROOT}/run/examples_perf_c2i_direct.exe"

REPEATS="${POISSON_STATIC_TDMA_REPEATS:-3}"
TIMED_ITERS="${POISSON_ITERS:-100}"
COARSE_ITERS="${POISSON_COARSE_ITERS:-30}"
WARMUP_ITERS="${POISSON_WARMUP:-5}"

export POISSON_CUDECOMP_BACKEND="${POISSON_CUDECOMP_BACKEND:-nccl_pl}"
export POISSON_TDMA_2RHS=1
export POISSON_TDMA_STATIC_OPERATOR=0
export POISSON_WRITE_SOLUTION=0
export POISSON_DEBUG_WARMUP="${POISSON_DEBUG_WARMUP:-5}"
export POISSON_DEBUG_COLD_PROFILE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"

for value_name in REPEATS TIMED_ITERS COARSE_ITERS WARMUP_ITERS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${value_name} must be a positive integer: ${value}"
        exit 3
    fi
done
if [[ "${SLURM_NTASKS}" -ne 4 || "${SLURM_NNODES}" -ne 2 ]]; then
    echo "[ERROR] Expected the MPMC two-node/four-rank allocation."
    exit 4
fi

mapfile -t ALLOCATED_NODES < <(
    scontrol show hostnames "${SLURM_JOB_NODELIST}"
)
if [[ "${#ALLOCATED_NODES[@]}" -ne 2 ]]; then
    echo "[ERROR] Expected exactly two allocated nodes."
    exit 5
fi
NODE0="${ALLOCATED_NODES[0]}"
NODE1="${ALLOCATED_NODES[1]}"

RESULT_ROOT="${PROJECT_ROOT}/results_static_tdma/job_${SLURM_JOB_ID}"
RANKFILE_ROOT="${RESULT_ROOT}/rankfiles"
mkdir -p "${RANKFILE_ROOT}"

# These are the three possible pair partitions of four ranks over two nodes.
# Testing all three avoids assuming how cuDecomp linearizes its 2x2 process
# grid.  Each node always receives two ranks and two GPUs.
cat > "${RANKFILE_ROOT}/pair01_23.rankfile" <<EOF
rank 0=${NODE0} slot=0
rank 1=${NODE0} slot=1
rank 2=${NODE1} slot=0
rank 3=${NODE1} slot=1
EOF
cat > "${RANKFILE_ROOT}/pair02_13.rankfile" <<EOF
rank 0=${NODE0} slot=0
rank 2=${NODE0} slot=1
rank 1=${NODE1} slot=0
rank 3=${NODE1} slot=1
EOF
cat > "${RANKFILE_ROOT}/pair03_12.rankfile" <<EOF
rank 0=${NODE0} slot=0
rank 3=${NODE0} slot=1
rank 1=${NODE1} slot=0
rank 2=${NODE1} slot=1
EOF

cat > "${RESULT_ROOT}/run_config.txt" <<EOF
JOB_ID=${SLURM_JOB_ID}
NODELIST=${SLURM_JOB_NODELIST}
NODE0=${NODE0}
NODE1=${NODE1}
NNODES=${SLURM_NNODES}
NTASKS=${SLURM_NTASKS}
PROJECT_ROOT=${PROJECT_ROOT}
CUDECOMP_BACKEND=${POISSON_CUDECOMP_BACKEND}
TDMA_2RHS=${POISSON_TDMA_2RHS}
NP_ORDER=1 2 4
NP4_PLACEMENTS=pair01_23 pair02_13 pair03_12
DATA_PATH=direct_C2I_to_FFT
TDMA_OPERATOR_VARIANTS=dynamic static
PERF_REPEATS=${REPEATS}
POISSON_WARMUP=${WARMUP_ITERS}
POISSON_ITERS=${TIMED_ITERS}
POISSON_COARSE_ITERS=${COARSE_ITERS}
POISSON_WRITE_SOLUTION=${POISSON_WRITE_SOLUTION}
EOF

echo "========================================"
echo "MPMC static Poisson-operator TDMA A/B and rank-placement sweep"
echo "========================================"
cat "${RESULT_ROOT}/run_config.txt"
echo "RMS is evaluated once for the final solution, outside timers."
echo "Unprofiled performance logs determine total seconds/solve."
echo "Coarse logs are diagnostic and use seven major boundaries."

echo
echo "========================================"
echo "Allocated GPUs"
echo "========================================"
printf '%s\n' "${ALLOCATED_NODES[@]}"
srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    bash -c '
        echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi -L
    '

mpi_command()
{
    local np="$1"
    local placement="$2"

    MPI_CMD=(
        "${MPIEXEC}"
        --prefix "${NVHPC_MPI_ROOT}"
        -np "${np}"
        --bind-to none
    )

    if [[ "${np}" -lt 4 ]]; then
        MPI_CMD+=(--map-by "ppr:${np}:node")
    else
        MPI_CMD+=(--rankfile "${RANKFILE_ROOT}/${placement}.rankfile")
    fi

    MPI_CMD+=(
        -x PATH
        -x LD_LIBRARY_PATH
        -x CUDA_DEVICE_ORDER
        -x POISSON_CUDECOMP_BACKEND
        -x POISSON_TDMA_2RHS
        -x POISSON_TDMA_STATIC_OPERATOR
        -x POISSON_WRITE_SOLUTION
        -x POISSON_WARMUP
        -x POISSON_ITERS
        -x POISSON_COARSE_PROFILE
        -x POISSON_DEBUG_WARMUP
        -x POISSON_DEBUG_COLD_PROFILE
        -x NCCL_DEBUG
        -x NCCL_DEBUG_SUBSYS
    )
}

run_one()
{
    local np="$1"
    local placement="$2"
    local operator="$3"
    local kind="$4"
    local repeat="$5"
    local case_root run_dir log_file input_file

    case "${kind}" in
        debug) EXECUTABLE="${DEBUG_DIRECT}" ;;
        perf|coarse) EXECUTABLE="${PERF_DIRECT}" ;;
        *)
            echo "[ERROR] Invalid run kind: ${kind}"
            exit 6
            ;;
    esac

    case "${operator}" in
        dynamic) export POISSON_TDMA_STATIC_OPERATOR=0 ;;
        static)  export POISSON_TDMA_STATIC_OPERATOR=1 ;;
        *)
            echo "[ERROR] Invalid TDMA operator variant: ${operator}"
            exit 7
            ;;
    esac

    case_root="${RESULT_ROOT}/np${np}/${placement}/${operator}"
    case "${kind}" in
        debug)
            run_dir="${case_root}/debug"
            log_file="debug_solver.log"
            export POISSON_COARSE_PROFILE=0
            export POISSON_WARMUP="${WARMUP_ITERS}"
            export POISSON_ITERS="${TIMED_ITERS}"
            ;;
        perf)
            run_dir="${case_root}/perf/repeat_${repeat}"
            log_file="perf_solver.log"
            export POISSON_COARSE_PROFILE=0
            export POISSON_WARMUP="${WARMUP_ITERS}"
            export POISSON_ITERS="${TIMED_ITERS}"
            ;;
        coarse)
            run_dir="${case_root}/coarse"
            log_file="coarse_solver.log"
            export POISSON_COARSE_PROFILE=1
            export POISSON_WARMUP="${WARMUP_ITERS}"
            export POISSON_ITERS="${COARSE_ITERS}"
            ;;
        *)
            echo "[ERROR] Invalid run kind: ${kind}"
            exit 8
            ;;
    esac

    input_file="${PROJECT_ROOT}/run/PARA_INPUT_${np}.dat"
    mkdir -p "${run_dir}"
    cp "${input_file}" "${run_dir}/"
    mpi_command "${np}" "${placement}"

    echo
    echo "============================================================"
    echo "np=${np} placement=${placement} TDMA_OPERATOR=${operator} kind=${kind} repeat=${repeat}"
    echo "============================================================"

    (
        cd "${run_dir}"
        "${MPI_CMD[@]}" "${EXECUTABLE}" 2>&1 | tee "${log_file}"
    )

    if ! grep -q 'RMS error against analytic solution:' \
        "${run_dir}/${log_file}"; then
        echo "[ERROR] Missing final RMS: np=${np} ${placement} ${operator} ${kind}"
        exit 9
    fi
    grep -q '\[C2I-BUILD\] direct:' "${run_dir}/${log_file}" || {
        echo "[ERROR] Direct C2I executable did not identify itself."
        exit 10
    }
    grep -q 'Stage-4 complex solve: shared operator + 2 RHS' \
        "${run_dir}/${log_file}" || {
            echo "[ERROR] Stage-4 2-RHS path was not selected."
            exit 11
        }
    if [[ "${operator}" == static ]]; then
        grep -q '\[PTDMA-STATIC\] Operator factors prepared before timed solves.' \
            "${run_dir}/${log_file}" || {
            echo "[ERROR] Static operator was not prepared."
            exit 12
        }
        grep -q 'Static operator apply: RHS-only kernels + 2 collectives' \
            "${run_dir}/${log_file}" || {
            echo "[ERROR] Static two-collective solve path was not selected."
            exit 13
        }
    else
        grep -q 'Dynamic operator apply: rebuild/factor + 5 collectives' \
            "${run_dir}/${log_file}" || {
            echo "[ERROR] Dynamic five-collective control path was not selected."
            exit 14
        }
    fi
    if [[ "${kind}" == coarse ]]; then
        grep -q 'Coarse performance phases' "${run_dir}/${log_file}" || {
            echo "[ERROR] Missing coarse phase report."
            exit 15
        }
    fi
    if find "${case_root}" -type f -name 'P_*.PLT' -print -quit \
        | grep -q .; then
        echo "[ERROR] Unexpected full solution dump."
        exit 16
    fi
}

record_rank_map()
{
    local np="$1"
    local placement="$2"

    export POISSON_COARSE_PROFILE=0
    export POISSON_WARMUP="${WARMUP_ITERS}"
    export POISSON_ITERS="${TIMED_ITERS}"
    mpi_command "${np}" "${placement}"

    "${MPI_CMD[@]}" bash -c '
        printf "host=%-8s world_rank=%s local_rank=%s CUDA_VISIBLE_DEVICES=%s\n" \
            "$(hostname)" \
            "${OMPI_COMM_WORLD_RANK:-unset}" \
            "${OMPI_COMM_WORLD_LOCAL_RANK:-unset}" \
            "${CUDA_VISIBLE_DEVICES:-unset}"
    ' | sort -k2 | tee "${RESULT_ROOT}/rank_map_np${np}_${placement}.log"
}

run_case()
{
    local np="$1"
    local placement="$2"
    local operator

    record_rank_map "${np}" "${placement}"

    for operator in dynamic static; do
        # Detailed debug is run for NP1/NP2 and the conventional NP4 mapping.
        if [[ "${np}" -lt 4 || "${placement}" == pair01_23 ]]; then
            run_one "${np}" "${placement}" "${operator}" debug 0
        fi
        for repeat in $(seq 1 "${REPEATS}"); do
            run_one "${np}" "${placement}" "${operator}" perf "${repeat}"
        done
        run_one "${np}" "${placement}" "${operator}" coarse 0
    done
}

run_case 1 default
run_case 2 default
for placement in pair01_23 pair02_13 pair03_12; do
    run_case 4 "${placement}"
done

"${PROJECT_ROOT}/summarize_static_tdma_sweep.sh" "${RESULT_ROOT}" \
    | tee "${RESULT_ROOT}/summary.txt"

echo
echo "[PASS] Static-operator TDMA A/B and NP4 rank-placement sweep completed."
echo "Results: ${RESULT_ROOT}"
