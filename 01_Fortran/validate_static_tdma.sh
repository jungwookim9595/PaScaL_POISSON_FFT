#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

fail()
{
    echo "[FAIL] $*" >&2
    exit 1
}

pass()
{
    echo "[PASS] $*"
}

scripts=(
    load_mpmc_env.sh
    build_static_tdma_mpmc.sh
    build_and_submit_mpmc_static_tdma.sh
    run_mpmc_static_tdma_sweep.slurm
    summarize_static_tdma_sweep.sh
    validate_static_tdma.sh
)

for script in "${scripts[@]}"; do
    bash -n "${script}" || fail "Shell syntax: ${script}"
done
pass "Shell syntax."

grep -q '^static-tdma:' makefile \
    || fail "Combined debug/performance build target is missing."
grep -q 'make -j "${BUILD_JOBS}" static-tdma' build_static_tdma_mpmc.sh \
    || fail "MPMC build script does not use the combined target."
pass "One-command debug/performance build."

ptdma="PaScaL_TDMA/src/pascal_tdma_cuda.f90"
poisson="src/fft_3d_poisson.f90"

grep -q 'PaScaL_TDMA_many_prepare_static_cuda' "${ptdma}" \
    || fail "Static operator prepare API is missing."
grep -q 'PaScaL_TDMA_many_solve_static_2rhs_cuda' "${ptdma}" \
    || fail "Static two-RHS apply API is missing."
grep -q 'PaScaL_TDMA_many_prepare_modified_static_cuda' "${ptdma}" \
    || fail "Coefficient-only modified-Thomas prepare kernel is missing."
grep -q 'PaScaL_TDMA_many_apply_modified_static_2rhs_cuda' "${ptdma}" \
    || fail "RHS-only modified-Thomas apply kernel is missing."
grep -q 'tdma_static_prepare_cuda' "${ptdma}" \
    || fail "Reduced-system factorization kernel is missing."
grep -q 'tdma_static_apply_2rhs_cuda' "${ptdma}" \
    || fail "Reduced-system RHS apply kernel is missing."
grep -q 'Repeated-solve collectives: 2' "${ptdma}" \
    || fail "Two-collective contract is not reported."
pass "Static local and reduced TDMA factorization/apply APIs."

grep -q 'POISSON_TDMA_STATIC_OPERATOR' "${poisson}" \
    || fail "Dynamic/static runtime selector is missing."
grep -q 'cuda_Poisson_TDMA_static_initial' "${poisson}" \
    || fail "Pre-iteration operator initializer is missing."
grep -q 'unpack the changing complex RHS' "${poisson}" \
    || fail "RHS-only Poisson packing path is missing."
grep -q 'cuda_ptdma_core_static_2rhs' "${poisson}" \
    || fail "Poisson static TDMA dispatch is missing."
grep -q 'cuda_Poisson_TDMA_static_initial' examples/main.f90 \
    || fail "Main initialization does not prepare the operator."
pass "Operator preparation occurs before Poisson iterations."

tmp_root="$(mktemp -d)"
cleanup()
{
    rm -rf "${tmp_root}"
}
trap cleanup EXIT

cpp -P \
    -DPOISSON_USE_CUDECOMP -DPERF_BUILD \
    -DPOISSON_COARSE_PROFILE -DPOISSON_DIRECT_C2I_FFT \
    -DDOUBLE_PRECISION "${poisson}" > "${tmp_root}/poisson.f90"
grep -Eq '^[[:space:]]*call PaScaL_TDMA_many_solve_static_2rhs_cuda' \
    "${tmp_root}/poisson.f90" \
    || fail "Preprocessed performance path lost static TDMA dispatch."
grep -Fq 'call fft_c2i_real_contiguous(PRHS_d, FFT_x1' \
    "${tmp_root}/poisson.f90" \
    || fail "Direct C2I-to-FFT optimization is missing."
pass "Preprocessed direct-C2I/static-TDMA performance path."

for placement in pair01_23 pair02_13 pair03_12; do
    grep -q "${placement}" run_mpmc_static_tdma_sweep.slurm \
        || fail "Missing NP4 placement: ${placement}"
done
grep -q -- '--rankfile' run_mpmc_static_tdma_sweep.slurm \
    || fail "OpenMPI rankfile launch is missing."
grep -q 'for operator in dynamic static' run_mpmc_static_tdma_sweep.slurm \
    || fail "Dynamic/static A/B loop is missing."
grep -q 'POISSON_TDMA_STATIC_OPERATOR' run_mpmc_static_tdma_sweep.slurm \
    || fail "Static operator selector is not exported to ranks."
grep -q 'POISSON_TDMA_2RHS=1' run_mpmc_static_tdma_sweep.slurm \
    || fail "Stage-4 two-RHS mode is not fixed on."
grep -q 'POISSON_WRITE_SOLUTION=0' run_mpmc_static_tdma_sweep.slurm \
    || fail "Full solution output is not disabled."
pass "MPMC dynamic/static A/B and exhaustive NP4 placement sweep."

for np_shape in '1:1,1,1' '2:1,1,2' '4:1,2,2'; do
    np="${np_shape%%:*}"
    shape="${np_shape#*:}"
    IFS=, read -r np1 np2 np3 <<< "${shape}"
    file="run/PARA_INPUT_${np}.dat"
    grep -Eq "^np1[[:space:]]*=[[:space:]]*${np1}" "${file}" \
        || fail "Wrong np1 in ${file}."
    grep -Eq "^np2[[:space:]]*=[[:space:]]*${np2}" "${file}" \
        || fail "Wrong np2 in ${file}."
    grep -Eq "^np3[[:space:]]*=[[:space:]]*${np3}" "${file}" \
        || fail "Wrong np3 in ${file}."
    grep -Eq '^ContinueFileout[[:space:]]*=[[:space:]]*\.false\.' "${file}" \
        || fail "Solution output is enabled in ${file}."
done
pass "Required 111 -> 112 -> 122 cube decompositions."

grep -q 'static_tdma_ab_comparison.csv' summarize_static_tdma_sweep.sh \
    || fail "Static TDMA A/B CSV summary is missing."
grep -q 'tdma_phase_ab_comparison.csv' summarize_static_tdma_sweep.sh \
    || fail "TDMA phase A/B CSV summary is missing."
grep -q 'Primary target NP4 < NP1' summarize_static_tdma_sweep.sh \
    || fail "NP4-vs-NP1 target report is missing."
grep -q 'Final RMS check runs once after all timers stop' examples/main.f90 \
    || fail "Final RMS timing separation is missing."
pass "RMS-only correctness and performance summaries."

pass "Static Poisson-operator TDMA package validation complete."
