#!/bin/bash

set -euo pipefail

resolve_project_root()
{
    local candidate

    for candidate in \
        "${PROJECT_ROOT:-}" \
        "${SLURM_SUBMIT_DIR:-}" \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
        if [[ -n "${candidate}" \
              && -f "${candidate}/src/fft_3d_poisson.f90" \
              && -f "${candidate}/vendor/cuDecomp/CMakeLists.txt" \
              && -f "${candidate}/makefile" \
              && -f "${candidate}/PaScaL_TDMA/src/pascal_tdma_cuda.f90" ]]; then
            cd "${candidate}"
            pwd
            return 0
        fi
    done
    return 1
}

if ! PROJECT_ROOT="$(resolve_project_root)"; then
    echo "[ERROR] Could not locate the static-TDMA project root."
    echo "Run this script from the extracted project directory."
    exit 2
fi
export PROJECT_ROOT
BUILD_JOBS="${BUILD_JOBS:-8}"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/load_mpmc_env.sh"

echo
echo "========================================"
echo "Build direct-C2I static-TDMA debug/performance executables"
echo "========================================"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "BUILD_JOBS=${BUILD_JOBS}"
echo "CPU target=-tp=px"
echo "GPU target=${MPMC_GPU_FLAGS}"
echo "CUDA toolkit=${NVHPC_CUDA_HOME}"

PROBE_ROOT="$(mktemp -d)"
cleanup_probe()
{
    rm -rf "${PROBE_ROOT}"
}
trap cleanup_probe EXIT

echo
echo "Checking NVHPC CUDA toolchain..."
nvfortran \
    -cuda \
    "${MPMC_GPU_FLAGS}" \
    -c "${PROJECT_ROOT}/src/para_range.f90" \
    -o "${PROBE_ROOT}/para_range.o" \
    -module "${PROBE_ROOT}"
echo "[PASS] NVHPC CUDA toolchain probe."

if [[ -z "${MPMC_CMAKE_BIN:-}" ]] \
    || ! run_mpmc_cmake --version >/dev/null 2>&1; then
    echo "[ERROR] A runnable CMake >= 3.16 is required for cuDecomp."
    echo 'If needed: export MPMC_CMAKE="$HOME/.local/cudecomp-cmake-env/bin/cmake"'
    exit 5
fi

CUDA_HOST_TAG="${MPMC_CUDA_HOST_VERSION//./_}"
CUDECOMP_BUILD="${PROJECT_ROOT}/_build/cudecomp_cuda${CUDA_VERSION}_cc80_gcc${CUDA_HOST_TAG}"
CUDECOMP_INSTALL="${PROJECT_ROOT}/_install/cudecomp"
mkdir -p "${CUDECOMP_BUILD}" "${CUDECOMP_INSTALL}"

echo
echo "========================================"
echo "Build vendored cuDecomp"
echo "========================================"
echo "source    : ${PROJECT_ROOT}/vendor/cuDecomp"
echo "build     : ${CUDECOMP_BUILD}"
echo "install   : ${CUDECOMP_INSTALL}"
echo "backend   : NCCL_PL runtime default"

run_mpmc_cmake \
    -S "${PROJECT_ROOT}/vendor/cuDecomp" \
    -B "${CUDECOMP_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CUDECOMP_INSTALL}" \
    -DCMAKE_CUDA_COMPILER="${MPMC_CUDA_HOME}/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="${MPMC_CUDA_HOST_CXX}" \
    -DCMAKE_CXX_COMPILER="${NVHPC_COMPILER_ROOT}/bin/nvc++" \
    -DCMAKE_Fortran_COMPILER="${NVHPC_COMPILER_ROOT}/bin/nvfortran" \
    -DMPI_CXX_COMPILER="${NVHPC_MPI_ROOT}/bin/mpicxx" \
    -DMPI_Fortran_COMPILER="${NVHPC_MPI_ROOT}/bin/mpif90" \
    -DNVHPC_CUDA_VERSION="${CUDA_VERSION}" \
    -DCUDECOMP_CUDA_CC_LIST=80 \
    -DCUDECOMP_BUILD_FORTRAN=ON \
    -DCUDECOMP_BUILD_EXTRAS=OFF \
    -DCUDECOMP_ENABLE_NVSHMEM=OFF \
    -DCUDECOMP_ENABLE_NVTX=OFF \
    -DCUDECOMP_NCCL_HOME="${CUDECOMP_NCCL_HOME}"

run_mpmc_cmake \
    --build "${CUDECOMP_BUILD}" --parallel "${BUILD_JOBS}"
run_mpmc_cmake --install "${CUDECOMP_BUILD}"

for required in \
    "${CUDECOMP_INSTALL}/include/cudecomp.mod" \
    "${CUDECOMP_INSTALL}/lib/libcudecomp.so" \
    "${CUDECOMP_INSTALL}/lib/libcudecomp_fort.so"; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing cuDecomp output: ${required}"
        exit 6
    fi
done

export LD_LIBRARY_PATH="${CUDECOMP_INSTALL}/lib:${LD_LIBRARY_PATH}"

make clean
make -j "${BUILD_JOBS}" static-tdma \
    CPU_ARCH_FLAGS=-tp=px \
    GPU_ARCH_FLAGS="${MPMC_GPU_FLAGS}" \
    CUDECOMP_HOME="${CUDECOMP_INSTALL}"

EXECUTABLES=(
    "${PROJECT_ROOT}/run/examples_debug_c2i_direct.exe"
    "${PROJECT_ROOT}/run/examples_perf_c2i_direct.exe"
)

for executable in "${EXECUTABLES[@]}"; do
    test -x "${executable}" || {
        echo "[ERROR] Missing executable: ${executable}"
        exit 7
    }
    if ldd "${executable}" | grep -q 'not found'; then
        echo "[ERROR] Unresolved library in ${executable}:"
        ldd "${executable}" | grep 'not found' || true
        exit 8
    fi
done

echo
echo "[PASS] MPMC debug/performance executables built."
printf '  %s\n' "${EXECUTABLES[@]}"
