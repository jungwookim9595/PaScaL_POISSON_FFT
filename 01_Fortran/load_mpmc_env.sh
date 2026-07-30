#!/bin/bash

# Source this file from build and Slurm scripts on the MPMC cluster.

if ! command -v module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    else
        echo "[ERROR] Environment Modules is not available."
        return 1 2>/dev/null || exit 1
    fi
fi

module purge
module load nvhpc/23.7

# Remove MPI installations inherited from an interactive shell before putting
# the OpenMPI bundled with NVHPC 23.7 first in PATH.
unset OPAL_PREFIX
unset OMPI_HOME
unset I_MPI_ROOT
unset MPI_HOME

export NVHPC_MPI_ROOT=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7/comm_libs/openmpi/openmpi-3.1.5
export NVHPC_COMPILER_ROOT=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7/compilers
export NVHPC_SDK_ROOT=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7

if [[ ! -x "${NVHPC_MPI_ROOT}/bin/mpirun" ]]; then
    echo "[ERROR] NVHPC OpenMPI was not found: ${NVHPC_MPI_ROOT}"
    return 1 2>/dev/null || exit 1
fi
if [[ ! -x "${NVHPC_COMPILER_ROOT}/bin/nvfortran" ]]; then
    echo "[ERROR] NVHPC compiler was not found: ${NVHPC_COMPILER_ROOT}"
    return 1 2>/dev/null || exit 1
fi

select_cuda_home()
{
    local candidate

    # The login node has no visible GPU, so NVHPC cannot select a toolkit from
    # the driver. Prefer the CUDA 12.2 toolkit bundled with HPC SDK 23.7.
    for candidate in \
        "${MPMC_CUDA_HOME:-}" \
        "${NVHPC_SDK_ROOT}/cuda/12.2" \
        "${NVHPC_SDK_ROOT}/cuda/11.8" \
        "${NVHPC_SDK_ROOT}/cuda" \
        "/usr/local/cuda-12.2" \
        "/usr/local/cuda-11.8" \
        "/usr/local/cuda"; do
        if [[ -n "${candidate}" \
              && -x "${candidate}/bin/nvcc" \
              && ( -f "${candidate}/include/cuda_runtime.h" \
                   || -f "${candidate}/targets/x86_64-linux/include/cuda_runtime.h" ) ]]; then
            cd "${candidate}"
            pwd
            return 0
        fi
    done

    return 1
}

if ! MPMC_CUDA_HOME="$(select_cuda_home)"; then
    echo "[ERROR] A complete CUDA toolkit compatible with NVHPC 23.7 was not found."
    echo "Checked HPC SDK CUDA 12.2/11.8 and /usr/local/cuda-*."
    echo "Inspect available toolkits with:"
    echo "  find ${NVHPC_SDK_ROOT}/cuda /usr/local -maxdepth 3 -type f -name nvcc 2>/dev/null"
    return 1 2>/dev/null || exit 1
fi
export MPMC_CUDA_HOME
export NVHPC_CUDA_HOME="${MPMC_CUDA_HOME}"
export NVCOMPILER_CUDA_HOME="${MPMC_CUDA_HOME}"
export CUDA_HOME="${MPMC_CUDA_HOME}"
export CUDA_PATH="${MPMC_CUDA_HOME}"

CUDA_VERSION="$(
    "${MPMC_CUDA_HOME}/bin/nvcc" --version \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -n 1
)"
if [[ -z "${CUDA_VERSION}" ]]; then
    echo "[ERROR] Could not determine CUDA version from ${MPMC_CUDA_HOME}/bin/nvcc."
    return 1 2>/dev/null || exit 1
fi
export CUDA_VERSION
export MPMC_GPU_FLAGS="-gpu=cc80,cuda${CUDA_VERSION}"

select_cuda_host_cxx()
{
    local candidate version major
    local -a candidates

    candidates=()
    [[ -n "${MPMC_CUDA_HOST_CXX:-}" ]] \
        && candidates+=("${MPMC_CUDA_HOST_CXX}")
    candidates+=(
        /usr/bin/g++-12
        /usr/bin/g++-11
        /usr/bin/g++-10
        /usr/bin/g++-9
        /usr/bin/g++
    )

    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] || continue
        version="$(
            "${candidate}" -dumpfullversion -dumpversion 2>/dev/null \
                | sed -n '1p'
        )"
        major="${version%%.*}"
        [[ "${major}" =~ ^[0-9]+$ ]] || continue
        if (( major >= 6 && major <= 12 )); then
            printf '%s|%s\n' "${candidate}" "${version}"
            return 0
        fi
    done
    return 1
}

if ! MPMC_CUDA_HOST_INFO="$(select_cuda_host_cxx)"; then
    echo "[ERROR] CUDA 12.2 requires a GNU C++ host compiler version <= 12."
    return 1 2>/dev/null || exit 1
fi
export MPMC_CUDA_HOST_CXX="${MPMC_CUDA_HOST_INFO%%|*}"
export MPMC_CUDA_HOST_VERSION="${MPMC_CUDA_HOST_INFO#*|}"
export CUDAHOSTCXX="${MPMC_CUDA_HOST_CXX}"
unset MPMC_CUDA_HOST_INFO

MATH_LIBS_CANDIDATE="${NVHPC_SDK_ROOT}/math_libs/${CUDA_VERSION}"
if [[ -d "${MATH_LIBS_CANDIDATE}" ]]; then
    export NVCOMPILER_MATH_LIBS_HOME="${MATH_LIBS_CANDIDATE}"
else
    unset NVCOMPILER_MATH_LIBS_HOME
fi

NCCL_CANDIDATES=(
    "${NVHPC_SDK_ROOT}/comm_libs/${CUDA_VERSION}/nccl"
    "${NVHPC_SDK_ROOT}/comm_libs/nccl"
)
for candidate in "${NCCL_CANDIDATES[@]}"; do
    if [[ -f "${candidate}/include/nccl.h" \
          && -f "${candidate}/lib/libnccl.so" ]]; then
        export CUDECOMP_NCCL_HOME="${candidate}"
        break
    fi
done
if [[ -z "${CUDECOMP_NCCL_HOME:-}" ]]; then
    echo "[ERROR] NCCL required by the Stage-4 nccl_pl control run was not found."
    return 1 2>/dev/null || exit 1
fi

export PATH="${NVHPC_MPI_ROOT}/bin:${NVHPC_COMPILER_ROOT}/bin:${PATH}"
export PATH="${MPMC_CUDA_HOME}/bin:${PATH}"

MPMC_LD_PATH="${NVHPC_MPI_ROOT}/lib:${NVHPC_COMPILER_ROOT}/lib"
if [[ -d "${MPMC_CUDA_HOME}/lib64" ]]; then
    MPMC_LD_PATH="${MPMC_LD_PATH}:${MPMC_CUDA_HOME}/lib64"
fi
if [[ -d "${MPMC_CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
    MPMC_LD_PATH="${MPMC_LD_PATH}:${MPMC_CUDA_HOME}/targets/x86_64-linux/lib"
fi
if [[ -n "${NVCOMPILER_MATH_LIBS_HOME:-}" \
      && -d "${NVCOMPILER_MATH_LIBS_HOME}/lib64" ]]; then
    MPMC_LD_PATH="${MPMC_LD_PATH}:${NVCOMPILER_MATH_LIBS_HOME}/lib64"
fi
if [[ -d "${CUDECOMP_NCCL_HOME}/lib" ]]; then
    MPMC_LD_PATH="${MPMC_LD_PATH}:${CUDECOMP_NCCL_HOME}/lib"
fi
if [[ -d "${CUDECOMP_NCCL_HOME}/lib64" ]]; then
    MPMC_LD_PATH="${MPMC_LD_PATH}:${CUDECOMP_NCCL_HOME}/lib64"
fi
export LD_LIBRARY_PATH="${MPMC_LD_PATH}:${LD_LIBRARY_PATH:-}"
export MPIEXEC="${NVHPC_MPI_ROOT}/bin/mpirun"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

hash -r

cmake_version_supported()
{
    local version_text="$1"
    local version major minor remainder

    version="$(
        sed -n '1s/.*version[[:space:]]\+\([0-9][0-9.]*\).*/\1/p' \
            <<< "${version_text}"
    )"
    version="${version%%$'\n'*}"
    major="${version%%.*}"
    remainder="${version#*.}"
    minor="${remainder%%.*}"

    [[ "${major}" =~ ^[0-9]+$ && "${minor}" =~ ^[0-9]+$ ]] || return 1
    (( major > 3 || (major == 3 && minor >= 16) ))
}

select_mpmc_cmake()
{
    local candidate detected runtime_dir output candidate_prefix
    local -a candidates runtime_dirs
    local -A seen

    candidates=()
    runtime_dirs=()

    [[ -n "${MPMC_CMAKE:-}" ]] && candidates+=("${MPMC_CMAKE}")
    candidates+=(/usr/bin/cmake /usr/local/bin/cmake)
    candidates+=("${HOME}/.local/cudecomp-cmake-env/bin/cmake")
    [[ -n "${CONDA_PREFIX:-}" ]] \
        && candidates+=("${CONDA_PREFIX}/bin/cmake")
    detected="$(command -v cmake 2>/dev/null || true)"
    [[ -n "${detected}" ]] && candidates+=("${detected}")

    [[ -n "${MPMC_CMAKE_LIBDIR:-}" ]] \
        && runtime_dirs+=("${MPMC_CMAKE_LIBDIR}")
    [[ -n "${CONDA_PREFIX:-}" ]] \
        && runtime_dirs+=("${CONDA_PREFIX}/lib")
    runtime_dirs+=(
        "${HOME}/.local/cudecomp-cmake-env/lib"
        "${HOME}/miniconda3/lib"
        "${HOME}/anaconda3/lib"
    )

    MPMC_CMAKE_BIN=""
    MPMC_CMAKE_RUNTIME_LIB=""
    MPMC_CMAKE_VERSION=""

    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" && -x "${candidate}" ]] || continue
        [[ -z "${seen[${candidate}]+x}" ]] || continue
        seen["${candidate}"]=1

        if output="$("${candidate}" --version 2>/dev/null)" \
            && cmake_version_supported "${output}"; then
            MPMC_CMAKE_BIN="${candidate}"
            MPMC_CMAKE_VERSION="$(
                sed -n '1s/.*version[[:space:]]\+//p' <<< "${output}"
            )"
            return 0
        fi

        candidate_prefix="$(cd "$(dirname "${candidate}")/.." && pwd)"
        for runtime_dir in \
            "${runtime_dirs[@]}" \
            "${candidate_prefix}/lib64" \
            "${candidate_prefix}/lib"; do
            [[ -f "${runtime_dir}/libstdc++.so.6" ]] || continue
            if output="$(
                env LD_LIBRARY_PATH="${runtime_dir}:${LD_LIBRARY_PATH:-}" \
                    "${candidate}" --version 2>/dev/null
            )" && cmake_version_supported "${output}"; then
                MPMC_CMAKE_BIN="${candidate}"
                MPMC_CMAKE_RUNTIME_LIB="${runtime_dir}"
                MPMC_CMAKE_VERSION="$(
                    sed -n '1s/.*version[[:space:]]\+//p' <<< "${output}"
                )"
                return 0
            fi
        done
    done
    return 1
}

run_mpmc_cmake()
{
    if [[ -z "${MPMC_CMAKE_BIN:-}" ]]; then
        echo "[ERROR] No executable CMake >= 3.16 was selected." >&2
        return 127
    fi
    if [[ -n "${MPMC_CMAKE_RUNTIME_LIB:-}" ]]; then
        env \
            LD_LIBRARY_PATH="${MPMC_CMAKE_RUNTIME_LIB}:${LD_LIBRARY_PATH:-}" \
            "${MPMC_CMAKE_BIN}" "$@"
    else
        "${MPMC_CMAKE_BIN}" "$@"
    fi
}

select_mpmc_cmake || true
export MPMC_CMAKE_BIN MPMC_CMAKE_RUNTIME_LIB MPMC_CMAKE_VERSION

echo "========================================"
echo "MPMC NVHPC environment"
echo "========================================"
echo "nvfortran : $(command -v nvfortran)"
echo "mpif90    : $(command -v mpif90)"
echo "mpifort   : $(command -v mpifort)"
echo "mpirun    : ${MPIEXEC}"
echo "CUDA home : ${NVHPC_CUDA_HOME}"
echo "CUDA ver  : ${CUDA_VERSION}"
echo "GPU flags : ${MPMC_GPU_FLAGS}"
echo "nvcc host : ${MPMC_CUDA_HOST_CXX} (GCC ${MPMC_CUDA_HOST_VERSION})"
echo "Math libs : ${NVCOMPILER_MATH_LIBS_HOME:-CUDA toolkit defaults}"
echo "NCCL home : ${CUDECOMP_NCCL_HOME}"
echo "cmake     : ${MPMC_CMAKE_BIN:-no working CMake >= 3.16 found}"
echo "cmake ver : ${MPMC_CMAKE_VERSION:-unavailable}"
